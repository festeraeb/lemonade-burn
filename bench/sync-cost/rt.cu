// per-sync floor micro-test: trivial kernel + 1.2 KiB read-back, N rounds, several ways
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static inline double now_us(){ return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define NF 300  // 1.2 KiB
__global__ void tiny(float* out, int seq){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<NF) out[i]=seq+i; }
// writes data then a sequence word last (zero-copy path)
__global__ void tiny_mapped(volatile float* out, volatile int* flag, int seq){ int i=threadIdx.x; for(int j=i;j<NF;j+=blockDim.x) out[j]=seq+j; __syncthreads(); __threadfence_system(); if(i==0) *flag=seq; }
__global__ void spin(float* out, long long cycles){ long long t0=clock64(); while(clock64()-t0<cycles){} if(threadIdx.x==0&&blockIdx.x==0) out[0]+=1.f; }
static void stats(const char* name, std::vector<double>& v){
  std::sort(v.begin(),v.end()); double s=0; for(double x:v) s+=x;
  printf("%-52s mean %7.2f  p50 %7.2f  p99 %7.2f  min %7.2f us\n",name,s/v.size(),v[v.size()/2],v[(size_t)(v.size()*0.99)],v[0]); fflush(stdout);
}
int main(int argc,char**argv){
  int N=argc>1?atoi(argv[1]):10000; int sched=argc>2?atoi(argv[2]):0; // 0 auto 1 spin 2 yield 3 blocking
  unsigned fl[]={cudaDeviceScheduleAuto,cudaDeviceScheduleSpin,cudaDeviceScheduleYield,cudaDeviceScheduleBlockingSync};
  CK(cudaSetDeviceFlags(fl[sched]|cudaDeviceMapHost));
  CK(cudaSetDevice(0));
  printf("# N=%d sched=%d\n",N,sched);
  float* d; CK(cudaMalloc(&d,4096)); CK(cudaMemset(d,0,4096));
  cudaStream_t s; CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
  float* pg=(float*)malloc(4096); float* pin; CK(cudaHostAlloc(&pin,4096,cudaHostAllocDefault));
  std::vector<double> v; v.reserve(N);
  for(int w=0;w<200;w++){ tiny<<<2,256,0,s>>>(d,w); } CK(cudaStreamSynchronize(s));
  // 0 issue cost
  { v.clear(); for(int i=0;i<N;i++){ double t=now_us(); tiny<<<2,256,0,s>>>(d,i); v.push_back(now_us()-t);} CK(cudaStreamSynchronize(s)); stats("0  launch call only (issue cost)",v); }
  // 1 launch + stream sync, no copy
  { v.clear(); for(int i=0;i<N;i++){ double t=now_us(); tiny<<<2,256,0,s>>>(d,i); CK(cudaStreamSynchronize(s)); v.push_back(now_us()-t);} stats("1  launch + streamSynchronize (no copy)",v); }
  // a pageable memcpy sync (blocking stream-ordered; cudaMemcpyAsync pageable + sync)
  { v.clear(); for(int i=0;i<N;i++){ double t=now_us(); tiny<<<2,256,0,s>>>(d,i); CK(cudaMemcpyAsync(pg,d,NF*4,cudaMemcpyDeviceToHost,s)); CK(cudaStreamSynchronize(s)); v.push_back(now_us()-t);} stats("a  launch + memcpyAsync pageable + streamSync",v); }
  { v.clear(); for(int i=0;i<N;i++){ double t=now_us(); tiny<<<2,256,0,s>>>(d,i); CK(cudaMemcpy(pg,d,NF*4,cudaMemcpyDeviceToHost)); v.push_back(now_us()-t);} CK(cudaDeviceSynchronize()); stats("a' launch + cudaMemcpy pageable (legacy stream)",v); }
  // b pinned
  { v.clear(); for(int i=0;i<N;i++){ double t=now_us(); tiny<<<2,256,0,s>>>(d,i); CK(cudaMemcpyAsync(pin,d,NF*4,cudaMemcpyDeviceToHost,s)); CK(cudaStreamSynchronize(s)); v.push_back(now_us()-t);} stats("b  launch + memcpyAsync pinned + streamSync",v); }
  // c pinned + async + event query spin
  { cudaEvent_t ev; CK(cudaEventCreateWithFlags(&ev,cudaEventDisableTiming)); v.clear(); for(int i=0;i<N;i++){ double t=now_us(); tiny<<<2,256,0,s>>>(d,i); CK(cudaMemcpyAsync(pin,d,NF*4,cudaMemcpyDeviceToHost,s)); CK(cudaEventRecord(ev,s)); while(cudaEventQuery(ev)==cudaErrorNotReady){} v.push_back(now_us()-t);} stats("c  launch + async pinned + event record + query spin",v);
    v.clear(); for(int i=0;i<N;i++){ double t=now_us(); tiny<<<2,256,0,s>>>(d,i); CK(cudaMemcpyAsync(pin,d,NF*4,cudaMemcpyDeviceToHost,s)); CK(cudaEventRecord(ev,s)); CK(cudaEventSynchronize(ev)); v.push_back(now_us()-t);} stats("c' launch + async pinned + event record + eventSync",v); }
  // d mapped zero-copy: kernel writes into host-mapped memory, host spins on flag
  { float* hp; CK(cudaHostAlloc(&hp,4096,cudaHostAllocMapped)); float* dp; CK(cudaHostGetDevicePointer(&dp,hp,0));
    int* hf; CK(cudaHostAlloc(&hf,64,cudaHostAllocMapped)); int* df; CK(cudaHostGetDevicePointer(&df,hf,0)); *hf=-1;
    v.clear(); for(int i=0;i<N;i++){ double t=now_us(); tiny_mapped<<<1,256,0,s>>>(dp,df,i); while(*(volatile int*)hf!=i){} v.push_back(now_us()-t);} CK(cudaStreamSynchronize(s)); stats("d  mapped zero-copy write + host flag spin (no sync)",v); }
  // chain test: 38 layers x (router k + read + K work kernels of S us)
  int K=8; double Sus=25; int layers=38; int rep=200;
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr,0)); long long cyc=(long long)(Sus*pr.clockRate/1000.0);
  float* e0; CK(cudaMalloc(&e0,64));
  for(int mode=0;mode<5;mode++){
    cudaEvent_t ev; CK(cudaEventCreateWithFlags(&ev,cudaEventDisableTiming));
    v.clear();
    for(int r=0;r<rep;r++){
      double t=now_us();
      for(int l=0;l<layers;l++){
        tiny<<<2,256,0,s>>>(d,l);             // "router"
        if(mode==1){ CK(cudaMemcpyAsync(pg,d,NF*4,cudaMemcpyDeviceToHost,s)); CK(cudaStreamSynchronize(s)); }
        if(mode==2){ CK(cudaMemcpyAsync(pin,d,NF*4,cudaMemcpyDeviceToHost,s)); CK(cudaStreamSynchronize(s)); }
        if(mode==3){ CK(cudaMemcpyAsync(pin,d,NF*4,cudaMemcpyDeviceToHost,s)); CK(cudaEventRecord(ev,s)); while(cudaEventQuery(ev)==cudaErrorNotReady){} }
        if(mode==4){ CK(cudaMemcpyAsync(pin,d,NF*4,cudaMemcpyDeviceToHost,s)); CK(cudaEventRecord(ev,s)); CK(cudaEventSynchronize(ev)); }
        for(int k=0;k<K;k++) spin<<<1,32,0,s>>>(e0,cyc);
      }
      CK(cudaStreamSynchronize(s)); v.push_back((now_us()-t)/layers);
    }
    const char* nm[]={"chain: no per-layer sync (floor of device+issue)","chain: pageable copy + streamSync each layer","chain: pinned copy + streamSync each layer","chain: pinned async + event query spin each layer","chain: pinned async + eventSynchronize each layer"};
    stats(nm[mode],v);
  }
  return 0;
}
