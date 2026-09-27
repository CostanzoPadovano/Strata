#include "strata/core/dual_expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "ggml.h"
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <random>
#include <cmath>
#include <cstdio>
#include <cstring>

int main(int argc,char** argv) {
    if(argc!=2)return 2;
    namespace fs=std::filesystem; namespace k=strata::kernels; namespace cpu=k::cpu;
    const fs::path dir=argv[1]; fs::create_directories(dir);
    const fs::path raw=dir/"synthetic-experts.raw";
    std::ofstream file(raw,std::ios::binary),index(dir/"native_experts.txt");
    std::mt19937 rng(109); std::normal_distribution<float> nd(0,0.025f);
    std::vector<float> w(2560*640),imatrix(2560,1.0f);
    std::vector<std::vector<uint8_t>> blobs;
    std::vector<cpu::NativeFmt> fmts;
    uint64_t file_at=0,logical=0;
    const int types[]={16,17,18,21,22,29,42};
    constexpr int NE=80;
    for(int l=0;l<7;++l) {
        cpu::NativeFmt f; std::string err;
        if(!cpu::native_fmt(types[l],l%2?42:20,2560,640,f,err)){std::fprintf(stderr,"%s\n",err.c_str());return 1;}
        fmts.push_back(f); blobs.emplace_back(2*f.bytes);
        uint64_t offsets[3]; size_t sizes[]={f.up_off,f.down_off-f.up_off,f.bytes-f.down_off};
        size_t at[]={0,f.up_off,f.down_off};
        for(int r=0;r<3;++r) {
            offsets[r]=file_at;
            for(int e=0;e<2;++e) {
                for(auto& v:w)v=nd(rng)*(e+1);
                auto* b=blobs.back().data()+e*f.bytes+at[r];
                const int nrow=r==2?2560:640, ncol=r==2?640:2560;
                const auto type=(ggml_type)(r==2?f.d_type:f.gu_type);
                const auto got=ggml_quantize_chunk(type,w.data(),b,0,nrow,ncol,imatrix.data());
                if(got!=sizes[r])return 1;
                file.write((const char*)b,(std::streamsize)sizes[r]); file_at+=sizes[r];
            }
            for(int e=2;e<NE;++e) {
                const auto* b=blobs.back().data()+(e%2)*f.bytes+at[r];
                file.write((const char*)b,(std::streamsize)sizes[r]);file_at+=sizes[r];
            }
        }
        index<<l<<' '<<f.gu_type<<' '<<f.d_type<<' '<<logical<<' '<<f.bytes<<' '
             <<offsets[0]<<' '<<offsets[1]<<' '<<offsets[2]<<'\n'; logical+=NE*f.bytes;
    }
    file.close();index.close();std::string err;
    if(!cpu::expert_layout_load(dir.string(),7,NE,err)){std::fprintf(stderr,"%s\n",err.c_str());return 1;}
    {
        strata::platform::DirectFile io;
        void* buffer=strata::platform::DirectFile::alloc_aligned(4096);
        if(!buffer || !io.open(raw.string(),err) || !io.submit(0,buffer,4096,91,err))return 1;
        // Deliberately leave a submitted completion undrained, cancel it, then reuse.
        if(!io.cancel_pending(5000) || !io.submit(0,buffer,4096,92,err))return 1;
        strata::platform::Completion c;
        if(io.wait(&c,1,5000)!=1 || c.tag!=92 || !c.ok || c.bytes!=4096)return 1;
        io.close();strata::platform::DirectFile::free_aligned(buffer);
    }
    strata::core::DualExpertSource src;
    if(!src.open(raw.string(),2ull<<30,34ull<<30,err)){std::fprintf(stderr,"%s\n",err.c_str());return 1;}
    if(src.host_bytes()!=0 || src.gpu_bytes()!=logical)return 1;
    int cases=0;
    for(int l=0;l<7;++l) {
        for(int e=0;e<2;++e) {
            if(std::memcmp(src.blob(l,e),blobs[l].data()+e*fmts[l].bytes,fmts[l].bytes))return 1;
            void* stage=nullptr;cudaMalloc(&stage,fmts[l].bytes);
            if(!src.stage_remote(l,e,stage,nullptr,err))return 1;
            std::vector<uint8_t> back(fmts[l].bytes);
            if(cudaMemcpy(back.data(),stage,back.size(),cudaMemcpyDeviceToHost)!=cudaSuccess)return 1;
            cudaFree(stage);
            if(std::memcmp(back.data(),blobs[l].data()+e*fmts[l].bytes,back.size()))return 1;
        }
        for(int nt : {1,2,3,4,8}) {
            const int n=nt*10;
            std::vector<float> x(nt*2560),got(n*2560),ref(n*2560);
            for(auto& v:x)v=nd(rng)*20;
            std::vector<int32_t> ids(n),start(n+1),dst(n),tok(n);
            for(int i=0;i<n;++i){ids[i]=(i+nt)%2;start[i]=dst[i]=i;tok[i]=i/10;}start[n]=n;
            if(!src.compute_remote(l,x.data(),ids.data(),nt,10,got.data(),err))return 1;
            int restored=-1;cudaGetDevice(&restored);if(restored!=0)return 1;
            void *dw=nullptr,*dx=nullptr,*dq=nullptr,*ds=nullptr,*dy=nullptr,*dp=nullptr,*db=nullptr,*dc=nullptr,*dd=nullptr,*dt=nullptr;
            std::vector<unsigned long long> ptr(n);
            auto alloc=[](void** p,size_t z){if(cudaMalloc(p,z)!=cudaSuccess)std::abort();};
            alloc(&dw,blobs[l].size());alloc(&dx,x.size()*4);alloc(&dq,nt*80*36);
            alloc(&ds,k::native_expert_scratch_bytes(n,640));alloc(&dy,got.size()*4);
            alloc(&dp,n*8);alloc(&db,(n+1)*4);alloc(&dc,4);alloc(&dd,n*4);alloc(&dt,n*4);
            for(int i=0;i<n;++i)ptr[i]=(unsigned long long)((uint8_t*)dw+ids[i]*fmts[l].bytes);
            auto copy=[](void* d,const void* h,size_t z){if(cudaMemcpy(d,h,z,cudaMemcpyHostToDevice)!=cudaSuccess)std::abort();};
            copy(dw,blobs[l].data(),blobs[l].size());copy(dx,x.data(),x.size()*4);copy(dp,ptr.data(),n*8);
            copy(db,start.data(),(n+1)*4);copy(dc,&n,4);copy(dd,dst.data(),n*4);copy(dt,tok.data(),n*4);
            k::quantize_q8_1_rows((float*)dx,nt,2560,dq,nullptr);
            k::native_expert_grouped(k::native_expert_layout(fmts[l].gu_type,fmts[l].d_type,2560,640),
               (unsigned long long*)dp,(int32_t*)db,(int32_t*)dc,(int32_t*)dd,(int32_t*)dt,n,n,dq,ds,(float*)dy,nullptr);
            if(cudaMemcpy(ref.data(),dy,ref.size()*4,cudaMemcpyDeviceToHost)!=cudaSuccess)return 1;
            for(void* p:{dw,dx,dq,ds,dy,dp,db,dc,dd,dt})cudaFree(p);
            if(std::memcmp(ref.data(),got.data(),got.size()*4)){std::fprintf(stderr,"GPU0/GPU1 mismatch layer%d T%d\n",l,nt);return 1;}
            double sum=0;for(float v:got){if(!std::isfinite(v))return 1;sum+=std::fabs(v);}
            if(sum==0)return 1;
            if(nt==1) {
                // Independent ggml float dequantization + scalar FP64 dot products.
                const auto& f=fmts[l];const auto* b=blobs[l].data()+ids[0]*f.bytes;
                std::vector<float> wr(2560),ff(640),want(2560);
                const auto* gu=ggml_get_type_traits((ggml_type)f.gu_type);
                const auto* down=ggml_get_type_traits((ggml_type)f.d_type);
                for(int r=0;r<640;++r) {
                    double gv=0,uv=0;gu->to_float(b+r*f.gu_row,wr.data(),2560);
                    for(int j=0;j<2560;++j)gv+=(double)wr[j]*x[j];
                    gu->to_float(b+f.up_off+r*f.gu_row,wr.data(),2560);
                    for(int j=0;j<2560;++j)uv+=(double)wr[j]*x[j];
                    ff[r]=(float)(gv/(1+std::exp(-gv))*uv);
                }
                double num=0,den=0;
                for(int r=0;r<2560;++r) {
                    down->to_float(b+f.down_off+r*f.d_row,wr.data(),640);double v=0;
                    for(int j=0;j<640;++j)v+=(double)wr[j]*ff[j];
                    num+=std::fabs((double)got[r]-v);den+=std::fabs(v);
                }
                std::printf("layout %d float reference relative L1 %.6f\n",f.gu_type,num/(den+1e-30));
                if(num/(den+1e-30)>0.02)return 1;
            }
            ++cases;
        }
    }
    float x[2560]={},out[2560*10]={};int32_t ids[10]={};ids[9]=NE;
    if(src.compute_remote(0,x,ids,1,10,out,err) || src.blob(-1,0) || src.blob(0,NE))return 1;
    strata::core::DualExpertSource compact;
    if(!compact.open(raw.string(),NE*fmts[0].bytes,34ull<<30,err))return 1;
    if(compact.host_bytes()+compact.gpu_bytes()!=logical || compact.host_bytes()==0)return 1;
    for(int l=0;l<7;++l)for(int e=0;e<2;++e)
        if(compact.pinned(l,e) || std::memcmp(compact.blob(l,e),blobs[l].data()+e*fmts[l].bytes,fmts[l].bytes))return 1;
    strata::core::DualExpertSource cold;
    if(!cold.open(raw.string(),NE*fmts[0].bytes,0,err) || cold.host_bytes()!=0 || cold.disk_bytes()==0)return 1;
    for(int repeat=0;repeat<12;++repeat)for(int l=0;l<7;++l) {
        int32_t routed[80];for(int i=0;i<80;++i)routed[i]=(i+repeat)%NE;
        cold.begin_layer(l,routed,80);
        const uint8_t* saved[80];
        for(int i=0;i<80;++i) {
            saved[i]=cold.blob(l,routed[i]);
            if(!saved[i] || std::memcmp(saved[i],blobs[l].data()+(routed[i]%2)*fmts[l].bytes,fmts[l].bytes))return 1;
        }
        if(!cold.remote_layer(l))for(int i=0;i<80;++i)
            if(std::memcmp(saved[i],blobs[l].data()+(routed[i]%2)*fmts[l].bytes,fmts[l].bytes))return 1;
    }
    const fs::path profile=dir/"synthetic-profile.bin";
    {
        std::ofstream p(profile,std::ios::binary);p.write("STRP",4);
        const uint32_t hdr[5]={1,7,NE,8,8};p.write((const char*)hdr,sizeof(hdr));
        for(int l:{1,2})for(int e=0;e<4;++e) {
            const uint16_t pair[2]={(uint16_t)l,(uint16_t)e};p.write((const char*)pair,sizeof(pair));
        }
    }
    strata::core::DualExpertSource reserved;
    const uint64_t primary_budget=4*((fmts[1].bytes+255)/256*256);
    if(!reserved.open(raw.string(),NE*fmts[0].bytes,34ull<<30,err,profile.string(),primary_budget))return 1;
    if(reserved.disk_bytes()!=4*fmts[1].bytes || reserved.host_bytes()+reserved.gpu_bytes()+reserved.disk_bytes()!=logical)return 1;
    for(int l=0;l<7;++l)for(int e=0;e<NE;++e)
        if(std::memcmp(reserved.blob(l,e),blobs[l].data()+(e%2)*fmts[l].bytes,fmts[l].bytes))return 1;
    std::printf("dual_expert_test OK: %d bit-identical GPU0/GPU1 cases, 7 quant layouts, float oracle, peer staging, duplicates, T1..8, changed inputs, bounds, compact host, direct cold reads and 80-entry lifetimes\n",cases);
    return 0;
}
