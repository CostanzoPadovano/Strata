// No GGUF: native IQ3 embedding gather, mapped source vs resident replica on the other GPU, in real graphs.
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void cu(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
constexpr int TYPE=21, N=2560, VOCAB=128, MAX_T=8, GUARD=16;
struct Device {
    int ordinal;
    void* table=nullptr;
    int32_t* tokens=nullptr;
    float* output=nullptr;
    cudaStream_t stream=nullptr;
    ~Device() {
        cudaSetDevice(ordinal);
        if(stream) cudaStreamSynchronize(stream);
        if(table) cudaFree(table);
        if(tokens) cudaFree(tokens);
        if(output) cudaFree(output);
        if(stream) cudaStreamDestroy(stream);
    }
    void initialize() {
        cu(cudaSetDevice(ordinal));
        cu(cudaMalloc(reinterpret_cast<void**>(&tokens),MAX_T*sizeof(int32_t)));
        cu(cudaMalloc(reinterpret_cast<void**>(&output),(MAX_T*N+2*GUARD)*sizeof(float)));
        cu(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    }
};
int main() {
    void* host=nullptr;
    try {
        int count=0;
        cu(cudaGetDeviceCount(&count));
        if(count<2) throw std::runtime_error("two visible CUDA devices required");
        for(int device:{0,1}) {cu(cudaSetDevice(device));cu(cudaFree(nullptr));}
        const size_t row=strata::kernels::iq_row_bytes(TYPE,N), bytes=row*VOCAB;
        // The installed IQ3_XXS model's embedding tensor is native type21 (IQ3_S): ten110-byte blocks.
        if(row!=1100) throw std::runtime_error("unexpected native IQ3_S row size");
        cu(cudaHostAlloc(&host,bytes,cudaHostAllocPortable|cudaHostAllocMapped));
        auto* data=static_cast<uint8_t*>(host);
        uint32_t state=12345;
        for(size_t i=0;i<bytes;++i) {state=state*1664525u+1013904223u;data[i]=uint8_t(state>>24);}
        for(int token=0;token<VOCAB;++token)
            for(size_t block=0;block<row;block+=110) {
                const uint16_t scale=uint16_t(0x3800+(token%64)*8);
                std::memcpy(data+token*row+block,&scale,sizeof(scale));
            }
        int cases=0;
        for(int direction=0;direction<2;++direction) {
            Device mapped{direction}, resident{1-direction};
            mapped.initialize();resident.initialize();
            cu(cudaSetDevice(mapped.ordinal));
            void* alias=nullptr;
            cu(cudaHostGetDevicePointer(&alias,host,0));
            cu(cudaSetDevice(resident.ordinal));
            cu(cudaMalloc(&resident.table,bytes));
            cu(cudaMemcpy(resident.table,host,bytes,cudaMemcpyHostToDevice));
            std::vector<uint32_t> reference(MAX_T*N+2*GUARD), actual(reference.size());
            for(int rows=1;rows<=MAX_T;++rows) {
                cudaGraphExec_t executions[2]{};
                Device* devices[2]={&mapped,&resident};
                for(int lane=0;lane<2;++lane) {
                    Device& device=*devices[lane];
                    cu(cudaSetDevice(device.ordinal));
                    cudaGraph_t graph=nullptr;
                    cu(cudaStreamBeginCapture(device.stream,cudaStreamCaptureModeThreadLocal));
                    strata::kernels::iq_embed_rows(TYPE,lane?resident.table:alias,row,device.tokens,rows,N,
                                                   device.output+GUARD,device.stream);
                    cu(cudaGetLastError());
                    cu(cudaStreamEndCapture(device.stream,&graph));
                    cu(cudaGraphInstantiate(&executions[lane],graph,nullptr,nullptr,0));
                    cu(cudaGraphDestroy(graph));
                }
                for(int rep=0;rep<4;++rep) {
                    int32_t ids[MAX_T];
                    for(int i=0;i<MAX_T;++i) ids[i]=(rows*17+rep*31+i*7)%VOCAB;
                    for(int lane=0;lane<2;++lane) {
                        Device& device=*devices[lane];
                        cu(cudaSetDevice(device.ordinal));
                        cu(cudaMemsetAsync(device.output,0xcd,actual.size()*4,device.stream));
                        cu(cudaMemcpyAsync(device.tokens,ids,sizeof(ids),cudaMemcpyHostToDevice,device.stream));
                        cu(cudaGraphLaunch(executions[lane],device.stream));
                        cu(cudaStreamSynchronize(device.stream));
                        auto& back=lane?actual:reference;
                        cu(cudaMemcpy(back.data(),device.output,back.size()*4,cudaMemcpyDeviceToHost));
                    }
                    if(reference!=actual) throw std::runtime_error("mapped/resident cross-device embedding bits differ");
                    for(int i=0;i<rows*N;++i) {
                        float value=0;
                        std::memcpy(&value,&actual[GUARD+i],sizeof(value));
                        if(actual[GUARD+i]==0xcdcdcdcdu || !std::isfinite(value))
                            throw std::runtime_error("embedding row not produced or non-finite");
                    }
                    for(size_t i=0;i<reference.size();++i)
                        if((i<GUARD || i>=GUARD+rows*N) && actual[i]!=0xcdcdcdcdu)
                            throw std::runtime_error("embedding gather wrote beyond selected rows");
                    ++cases;
                }
                for(int lane=0;lane<2;++lane) {cu(cudaSetDevice(devices[lane]->ordinal));cu(cudaGraphExecDestroy(executions[lane]));}
            }
        }
        cu(cudaFreeHost(host));host=nullptr;
        std::printf("PASS %d embedding graph bit checks: mapped/resident, both directions, T1..8; no GGUF\n",cases);
        return 0;
    } catch(const std::exception& error) {
        if(host) cudaFreeHost(host);
        std::fprintf(stderr,"FAIL embedding graph parity: %s\n",error.what());
        return 1;
    }
}
