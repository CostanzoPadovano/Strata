#include "strata/core/dual_expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <map>

namespace strata::core {
namespace {
constexpr int H=2560, FF=640, CAP=80, MAX_T=8;
constexpr uint64_t GiB=1ull<<30;
struct DeviceScope {
    int old=0;
    explicit DeviceScope(int d) { cudaGetDevice(&old); cudaSetDevice(d); }
    ~DeviceScope() { cudaSetDevice(old); }
};
bool cu(cudaError_t e, std::string& err) {
    if(e==cudaSuccess) return true;
    err=std::string("GPU1 expert lane: ")+cudaGetErrorString(e); return false;
}
}
DualExpertSource::~DualExpertSource() {
    DeviceScope scope(device_);
    if(stream_) cudaStreamSynchronize((cudaStream_t)stream_);
    for(auto& graph:graphs_)cudaGraphExecDestroy((cudaGraphExec_t)graph.second);
    for(void* p : {hx_,hp_,hb_,hc_,hd_,ht_,hy_})if(p)cudaFreeHost(p);
    for(void* p : {gpu_,x_,xq_,scratch_,y_,ptr_,start_,count_,dst_,tok_}) if(p) cudaFree(p);
    if(stream_) cudaStreamDestroy((cudaStream_t)stream_);
    files_.clear(); // cancellation completion must precede freeing DMA read storage
    strata::platform::DirectFile::free_aligned(io_buffer_);
}
bool DualExpertSource::remote_layer(int64_t l) const {
    return l>=0 && (size_t)l<remote_.size() && remote_[(size_t)l];
}
bool DualExpertSource::open(const std::string& gguf, uint64_t gpu_budget, uint64_t host_budget, std::string& err,
                            const std::string& profile,uint64_t primary_cache_budget) {
    const auto& lay=strata::kernels::cpu::expert_layout();
    int devices=0;
    if(!cu(cudaGetDeviceCount(&devices),err) || devices<2 || !lay.native || lay.gguf_off.size()!=size_t(lay.n_layers*3)
       || gpu_budget>12*GiB || host_budget>34*GiB || gpu_budget==0 || primary_cache_budget>6*GiB) {
        err="GPU1 lane requires two GPUs, native GGUF, GPU budget 1..12 GiB and host budget <=34 GiB"; return false;
    }
    remote_.assign((size_t)lay.n_layers,false);
    host_off_.assign((size_t)(lay.n_layers*lay.n_expert),-1); gpu_off_.resize(remote_.size());
    uint64_t cpu_total=0;
    // Whole layers, in deterministic order; no eviction, duplication or runtime growth.
    for(int64_t l=0;l<lay.n_layers;++l) {
        const auto& fmt=lay.fmt[(size_t)l];
        if((fmt.gu_type!=16 && fmt.gu_type!=17 && fmt.gu_type!=18 && fmt.gu_type!=21 &&
            fmt.gu_type!=22 && fmt.gu_type!=29 && fmt.gu_type!=42) || (fmt.d_type!=20 && fmt.d_type!=42)) {
            err="Unsupported native expert quantization in GPU1 lane";return false;
        }
        uint64_t bytes=lay.blob_bytes(l)*(uint64_t)lay.n_expert;
        if(bytes<=gpu_budget-gpu_bytes_) {
            remote_[(size_t)l]=true; gpu_off_[(size_t)l]=gpu_bytes_; gpu_bytes_+=bytes;
        } else { cpu_total+=bytes; }
    }
    if(!gpu_bytes_) {err="No layer fits the GPU1 budget";return false;}
    std::vector<std::pair<int32_t,int32_t>> ranked;int64_t slots=0;
    if(!profile.empty() && !read_expert_profile(profile,lay.n_layers,lay.n_expert,ranked,slots,err))return false;
    std::vector<bool> primary(host_off_.size(),false);uint64_t primary_bytes=0;
    for(const auto& pr:ranked) {
        if(remote_layer(pr.first))continue;
        const auto bytes=(lay.blob_bytes(pr.first)+255)/256*256;
        if(bytes>primary_cache_budget-primary_bytes)break;
        primary_bytes+=bytes;primary[(size_t)(pr.first*lay.n_expert+pr.second)]=true;
    }
    // Fixed RAM residency, ranked first by the same immutable expert profile.
    // Cold payloads remain in the GGUF, never mmap; only an 80-entry window is read.
    auto retain=[&](int64_t l,int64_t e) {
        const size_t key=(size_t)(l*lay.n_expert+e);
        const uint64_t bytes=lay.blob_bytes(l);
        if(!remote_layer(l) && !primary[key] && host_off_[key]<0 && bytes<=host_budget-host_bytes_) {
            host_off_[key]=(int64_t)host_bytes_;host_bytes_+=bytes;
        }
    };
    for(const auto& pr:ranked)retain(pr.first,pr.second);
    for(int64_t l=0;l<lay.n_layers;++l)for(int64_t e=0;e<lay.n_expert;++e)retain(l,e);
    disk_bytes_=cpu_total-host_bytes_;
    std::map<std::string,size_t> file_index;file_of_.resize(remote_.size());
    for(int64_t l=0;l<lay.n_layers;++l) {
        std::filesystem::path path(gguf);
        if(!lay.gguf_file.empty() && !lay.gguf_file[(size_t)l].empty())path=path.parent_path()/lay.gguf_file[(size_t)l];
        auto found=file_index.find(path.string());
        if(found==file_index.end()) {
            auto f=std::make_unique<strata::platform::DirectFile>();
            if(!f->open(path.string(),err))return false;
            const size_t at=files_.size();files_.push_back(std::move(f));file_index[path.string()]=at;file_of_[(size_t)l]=at;
        } else file_of_[(size_t)l]=found->second;
    }
    io_buffer_=strata::platform::DirectFile::alloc_aligned((size_t)lay.max_blob+8192);
    if(!io_buffer_) {err="Cannot allocate bounded direct-read scratch";return false;}
    {
        DeviceScope scope(device_);
        size_t free_b=0,total_b=0;
        if(!cu(cudaMemGetInfo(&free_b,&total_b),err) || free_b<gpu_bytes_+2*GiB+(64ull<<20)) {
            err="GPU1 fixed experts would cross the 2 GiB reserve";return false;
        }
        std::fprintf(stderr,"strata dual: allocating GPU1 experts %.3f GiB\n",(double)gpu_bytes_/GiB);
        if(!cu(cudaMalloc(&gpu_,(size_t)gpu_bytes_),err)) {err="allocating GPU1 experts: "+err;return false;}
        cudaStream_t stream=nullptr;
        if(!cu(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),err)) return false;
        stream_=stream;
        for(auto a : {std::pair<void**,size_t>{&x_,MAX_T*H*4}, {&xq_,MAX_T*(H/32)*36},
             {&scratch_,strata::kernels::native_expert_scratch_bytes(CAP,FF)}, {&y_,CAP*H*4},
             {&ptr_,CAP*8}, {&start_,(CAP+1)*4}, {&count_,4}, {&dst_,CAP*4}, {&tok_,CAP*4}})
            if(!cu(cudaMalloc(a.first,a.second),err)) {err="allocating GPU1 scratch: "+err;return false;}
        for(auto a : {std::pair<void**,size_t>{&hx_,MAX_T*H*4},{&hp_,CAP*8},{&hb_,(CAP+1)*4},
             {&hc_,4},{&hd_,CAP*4},{&ht_,CAP*4},{&hy_,CAP*H*4}})
            if(!cu(cudaHostAlloc(a.first,a.second,cudaHostAllocPortable),err))return false;
    }
    if(host_bytes_) {
        host_=std::make_unique<PinnedArena>(host_bytes_,std::vector<uint64_t>{},false);
        if(!host_->valid()) {err="Cannot allocate compact host expert arena";return false;}
        std::fprintf(stderr,"strata dual: host arena: %s\n",host_->note.c_str());
    }
    bounce_.resize((size_t)lay.max_blob);
    if(disk_bytes_) {cold_.resize(CAP*(size_t)lay.max_blob);cold_keys_.assign(CAP,-1);}
    for(int64_t l=0;l<lay.n_layers;++l) {
        const auto& f=lay.fmt[(size_t)l];
        for(int64_t e=0;e<lay.n_expert;++e) {
            const auto off=host_off_[(size_t)(l*lay.n_expert+e)];
            if(!remote_layer(l) && off<0)continue;
            uint8_t* b=remote_layer(l)?bounce_.data():host_->data()+off;
            if(!read_blob(l,e,b,err))return false;
            if(remote_layer(l)) {
                DeviceScope scope(device_);
                if(!cu(cudaMemcpy((uint8_t*)gpu_+gpu_off_[(size_t)l]+e*f.bytes,b,f.bytes,cudaMemcpyHostToDevice),err)) {
                    err="copying GPU1 layer "+std::to_string(l)+" expert "+std::to_string(e)+": "+err;return false;
                }
            }
        }
        std::fprintf(stderr,"strata dual: layer %lld -> %s\n",(long long)l,remote_layer(l)?"GPU1":"bounded CPU RAM / direct SSD");
    }
    std::fprintf(stderr,"strata dual: %.3f GiB host + %.3f GiB GPU1 + %.3f GiB direct SSD; GPU1 experts have NO host duplicate\n",
                 (double)host_bytes_/GiB,(double)gpu_bytes_/GiB,(double)disk_bytes_/GiB);
    std::fprintf(stderr,"strata dual: %.3f GiB of GPU0 cache payloads excluded from CPU hot set (direct SSD fallback when cache lent)\n",(double)primary_bytes/GiB);
    return true;
}
bool DualExpertSource::read_blob(int64_t l,int64_t e,uint8_t* dst,std::string& err) {
    if(io_failed_) {err="Expert I/O is latched failed; refusing reuse";return false;}
    const auto& lay=strata::kernels::cpu::expert_layout();const auto& f=lay.fmt[(size_t)l];
    auto& file=*files_[file_of_[(size_t)l]];
    const uint64_t sizes[3]={f.up_off,f.down_off-f.up_off,f.bytes-f.down_off};
    const uint64_t in_blob[3]={0,f.up_off,f.down_off};
    for(int r=0;r<3;++r) {
        const uint64_t off=lay.gguf_off[(size_t)l*3+r]+(uint64_t)e*sizes[r];
        if(off>file.size() || sizes[r]>file.size()-off) {err="Expert GGUF slice out of bounds";return false;}
        const uint64_t aligned=off&~4095ull,inside=off-aligned;
        const auto bytes=(uint32_t)((inside+sizes[r]+4095)&~4095ull);
        if(bytes>(uint64_t)lay.max_blob+8192 || !file.submit(aligned,io_buffer_,bytes,0,err))return false;
        strata::platform::Completion c;
        if(file.wait(&c,1,10000)!=1 || !c.ok || c.bytes<inside+sizes[r]) {
            io_failed_=true;(void)file.cancel_pending(5000);
            err="Bounded direct expert read failed/timed out";return false;
        }
        std::memcpy(dst+in_blob[r],(uint8_t*)io_buffer_+inside,(size_t)sizes[r]);
    }
    return true;
}
void DualExpertSource::begin_layer(int64_t,const int32_t*,int64_t) {
    std::fill(cold_keys_.begin(),cold_keys_.end(),-1);cold_next_=0;
}
const uint8_t* DualExpertSource::blob(int64_t l,int64_t e) {
    const auto& lay=strata::kernels::cpu::expert_layout();
    if(l<0 || l>=lay.n_layers || e<0 || e>=lay.n_expert || (host_bytes_ && !host_))return nullptr;
    const auto bytes=lay.blob_bytes(l);
    if(!remote_layer(l)) {
        const int64_t key=l*lay.n_expert+e,off=host_off_[(size_t)key];
        if(off>=0)return host_->data()+off;
        for(size_t i=0;i<cold_keys_.size();++i)if(cold_keys_[i]==key)return cold_.data()+i*lay.max_blob;
        if(cold_keys_.empty())return nullptr;
        const size_t slot=cold_next_++%CAP;uint8_t* b=cold_.data()+slot*lay.max_blob;std::string err;
        if(!read_blob(l,e,b,err)) {std::fprintf(stderr,"strata dual: %s\n",err.c_str());return nullptr;}
        cold_keys_[slot]=key;return b;
    }
    DeviceScope scope(device_);
    if(cudaMemcpy(bounce_.data(),(uint8_t*)gpu_+gpu_off_[(size_t)l]+e*bytes,(size_t)bytes,cudaMemcpyDeviceToHost)!=cudaSuccess)return nullptr;
    return bounce_.data();
}
bool DualExpertSource::pinned(int64_t l,int64_t e) const {
    (void)l;(void)e;return false; // Prefill owns the only CUDA-pinned host staging.
}
bool DualExpertSource::stage_remote(int64_t l,int64_t e,void* dst,void* stream,std::string& err) {
    const auto& lay=strata::kernels::cpu::expert_layout();
    if(!remote_layer(l) || e<0 || e>=lay.n_expert || !dst) {err="GPU1 staging index out of bounds";return false;}
    // Works without NVLink/P2P access: CUDA uses its bounded copy path when peer access is unavailable.
    return cu(cudaMemcpyPeerAsync(dst,0,(uint8_t*)gpu_+gpu_off_[(size_t)l]+e*lay.blob_bytes(l),
                device_,(size_t)lay.blob_bytes(l),(cudaStream_t)stream),err);
}
bool DualExpertSource::compute_remote(int64_t l,const float* x,const int32_t* ids,int64_t nt,int64_t k,float* out,std::string& err) {
    if(!remote_layer(l) || nt<1 || nt>MAX_T || k!=10 || nt*k>CAP) {err="GPU1 expert window out of bounds";return false;}
    const auto& lay=strata::kernels::cpu::expert_layout();
    const auto& f=lay.fmt[(size_t)l];
    unsigned long long ptr[CAP]; int32_t start[CAP+1],dst[CAP],tok[CAP];
    // One group per entry is deliberately simple. Duplicate routed ids remain separate output rows.
    const int32_t n=(int32_t)(nt*k);
    for(int i=0;i<n;++i) {
        if(ids[i]<0 || ids[i]>=lay.n_expert){err="GPU1 expert id out of bounds";return false;}
        ptr[i]=(unsigned long long)((uint8_t*)gpu_+gpu_off_[(size_t)l]+ids[i]*f.bytes);
        start[i]=i; dst[i]=i; tok[i]=(int32_t)(i/k);
    }
    start[n]=n;
    DeviceScope scope(device_); auto s=(cudaStream_t)stream_;
    std::memcpy(hx_,x,(size_t)nt*H*4);std::memcpy(hp_,ptr,n*8);std::memcpy(hb_,start,(n+1)*4);
    std::memcpy(hc_,&n,4);std::memcpy(hd_,dst,n*4);std::memcpy(ht_,tok,n*4);
    const uint32_t key=(uint32_t)(f.gu_type<<24|f.d_type<<16|nt);
    auto found=graphs_.find(key);
    if(found==graphs_.end()) {
        if(graphs_.size()>=112){err="GPU1 graph bound exceeded";return false;}
        if(!cu(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal),err))return false;
        auto copy=[&](void* d,const void* h,size_t bytes){return cu(cudaMemcpyAsync(d,h,bytes,cudaMemcpyHostToDevice,s),err);};
        bool ok=copy(x_,hx_,(size_t)nt*H*4) && copy(ptr_,hp_,n*8) && copy(start_,hb_,(n+1)*4) &&
                copy(count_,hc_,4) && copy(dst_,hd_,n*4) && copy(tok_,ht_,n*4);
        if(ok) {
            strata::kernels::quantize_q8_1_rows((const float*)x_,nt,H,xq_,s);
            const auto L=strata::kernels::native_expert_layout(f.gu_type,f.d_type,H,FF);
            strata::kernels::native_expert_grouped(L,(const unsigned long long*)ptr_,(const int32_t*)start_,
                (const int32_t*)count_,(const int32_t*)dst_,(const int32_t*)tok_,n,n,xq_,scratch_,(float*)y_,s);
            ok=cu(cudaGetLastError(),err) && cu(cudaMemcpyAsync(hy_,y_,(size_t)n*H*4,cudaMemcpyDeviceToHost,s),err);
        }
        cudaGraph_t graph=nullptr;const auto ended=cudaStreamEndCapture(s,&graph);
        if(!ok || !cu(ended,err)) {if(graph)cudaGraphDestroy(graph);return false;}
        cudaGraphExec_t exec=nullptr;const bool made=cu(cudaGraphInstantiate(&exec,graph,0),err);cudaGraphDestroy(graph);
        if(!made)return false;
        found=graphs_.emplace(key,(void*)exec).first;
    }
    if(!cu(cudaGraphLaunch((cudaGraphExec_t)found->second,s),err) || !cu(cudaStreamSynchronize(s),err))return false;
    std::memcpy(out,hy_,(size_t)n*H*4);return true;
}
}
