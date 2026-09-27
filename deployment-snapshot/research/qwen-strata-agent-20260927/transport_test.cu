// Bounded no-GGUF test of the same staged transport used by remote MTP.
#include "strata/core/mtp.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void cu(cudaError_t e) {
    if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static constexpr size_t ROW = 4 * 2560, MAX_N = 8 * ROW, GUARD = 16;
__global__ void produce(uint32_t* x, size_t n, uint32_t salt) {
    const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]=0x3f000000u ^ (uint32_t(i)*2654435761u + salt);
}
__global__ void transform(uint32_t* x, size_t n) {
    const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]^=0x01020304u;
}
struct Lane {
    int src, dst;
    uint32_t *a=nullptr, *b=nullptr;
    void* host=nullptr;
    cudaStream_t producer=nullptr, consumer=nullptr;
    ~Lane() {
        cudaSetDevice(src);
        if(producer) cudaStreamSynchronize(producer);
        if(a) cudaFree(a);
        if(producer) cudaStreamDestroy(producer);
        cudaSetDevice(dst);
        if(consumer) cudaStreamSynchronize(consumer);
        if(b) cudaFree(b);
        if(consumer) cudaStreamDestroy(consumer);
        if(host) cudaFreeHost(host);
    }
};
int main() {
    try {
        int count=0;cu(cudaGetDeviceCount(&count));
        if(count<2) throw std::runtime_error("two visible devices required");
        uint64_t bytes=0;int cases=0;
        for(int direction=0;direction<2;++direction) {
            Lane lane{direction,1-direction};
            int peer=-1;cu(cudaDeviceCanAccessPeer(&peer,lane.src,lane.dst));
            std::printf("peer %d->%d = %d; explicit host staging regardless\n",lane.src,lane.dst,peer);
            cu(cudaSetDevice(lane.src));
            cu(cudaMalloc((void**)&lane.a,MAX_N*4));
            cu(cudaStreamCreateWithFlags(&lane.producer,cudaStreamNonBlocking));
            cu(cudaHostAlloc(&lane.host,MAX_N*4,cudaHostAllocPortable));
            cu(cudaSetDevice(lane.dst));
            cu(cudaMalloc((void**)&lane.b,(MAX_N+2*GUARD)*4));
            cu(cudaStreamCreateWithFlags(&lane.consumer,cudaStreamNonBlocking));
            std::vector<uint32_t> back(MAX_N+2*GUARD);
            for(int t=1;t<=8;++t) {
                const size_t n=ROW*t;
                cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
                cu(cudaStreamBeginCapture(lane.consumer,cudaStreamCaptureModeThreadLocal));
                transform<<<(unsigned)((n+255)/256),256,0,lane.consumer>>>(lane.b+GUARD,n);
                cu(cudaGetLastError());
                cu(cudaStreamEndCapture(lane.consumer,&graph));
                cu(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
                cu(cudaGraphDestroy(graph));
                for(int rep=0;rep<4;++rep) {
                    const uint32_t salt=(uint32_t)(t*31+rep);
                    cu(cudaSetDevice(lane.src));
                    produce<<<(unsigned)((n+255)/256),256,0,lane.producer>>>(lane.a,n,salt);
                    cu(cudaGetLastError());
                    // Required ordering against the nonblocking source stream.
                    cu(cudaStreamSynchronize(lane.producer));
                    cu(cudaSetDevice(lane.dst));
                    // A default-stream memset need not order against this
                    // nonblocking consumer. Keep reset/copy/graph on one stream.
                    cu(cudaMemsetAsync(lane.b,0xcd,(MAX_N+2*GUARD)*4,lane.consumer));
                    const int caller=(rep%2)?lane.src:lane.dst;
                    cu(cudaSetDevice(caller));
                    std::string err;
                    if(!strata::core::mtp_copy_device_staged(lane.b+GUARD,lane.dst,lane.a,lane.src,
                                                            lane.host,n*4,lane.consumer,err))
                        throw std::runtime_error(err);
                    int after=-1;cu(cudaGetDevice(&after));
                    if(after!=caller) throw std::runtime_error("transport failed caller-device restoration");
                    cu(cudaSetDevice(lane.dst));
                    cu(cudaGraphLaunch(exec,lane.consumer));
                    cu(cudaStreamSynchronize(lane.consumer));
                    cu(cudaMemcpy(back.data(),lane.b,back.size()*4,cudaMemcpyDeviceToHost));
                    for(size_t i=0;i<GUARD;++i)
                        if(back[i]!=0xcdcdcdcdu || back[MAX_N+GUARD+i]!=0xcdcdcdcdu)
                            throw std::runtime_error("transport/graph crossed guard");
                    for(size_t i=0;i<n;++i) {
                        const uint32_t expected=(0x3f000000u ^ (uint32_t(i)*2654435761u+salt)) ^ 0x01020304u;
                        if(back[GUARD+i]!=expected) {
                            std::fprintf(stderr,"mismatch %d->%d T%d rep%d index%llu got%08x expected%08x\n",
                                         lane.src,lane.dst,t,rep,(unsigned long long)i,back[GUARD+i],expected);
                            throw std::runtime_error("staged graph bit mismatch");
                        }
                    }
                    for(size_t i=n;i<MAX_N;++i)
                        if(back[GUARD+i]!=0xcdcdcdcdu) throw std::runtime_error("write beyond requested rows");
                    bytes+=n*4;++cases;
                }
                cu(cudaGraphExecDestroy(exec));
            }
        }
        std::printf("PASS %d staged copies; %llu bytes; two GPU kernels; T1..8; graphs/guards/device restoration\n",
                    cases,(unsigned long long)bytes);
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"FAIL staged MTP transport: %s\n",e.what());return 1;
    }
}
