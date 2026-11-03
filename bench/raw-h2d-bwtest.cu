#include <cstdio>
#include <cuda_runtime.h>
static double run(void* h, void* d, size_t n, int iters){
  cudaMemcpy(d,h,n,cudaMemcpyHostToDevice); cudaDeviceSynchronize();
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a);
  for(int i=0;i<iters;i++) cudaMemcpy(d,h,n,cudaMemcpyHostToDevice);
  cudaEventRecord(b); cudaEventSynchronize(b);
  float ms=0; cudaEventElapsedTime(&ms,a,b);
  return (double)n*iters/1e6/(ms/1e3);           // MB/s
}
int main(){
  int nd=0; cudaGetDeviceCount(&nd);
  size_t sizes[2]={5308416ULL,67108864ULL};
  for(int dev=0;dev<nd;dev++){
    cudaSetDevice(dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p,dev);
    int gen=0,wid=0;
    cudaDeviceGetAttribute(&gen,cudaDevAttrPciBusId,dev);
    printf("dev %d  %s\n",dev,p.name);
    for(int s=0;s<2;s++){
      size_t n=sizes[s]; void*d=nullptr; cudaMalloc(&d,n);
      void*hp=nullptr; cudaMallocHost(&hp,n);          // pinned
      void*hu=malloc(n);                                // pageable
      printf("   %6.1f MB  pinned %8.0f MB/s   pageable %8.0f MB/s\n",
             n/1e6, run(hp,d,n,30), run(hu,d,n,30));
      cudaFree(d); cudaFreeHost(hp); free(hu);
    }
  }
  return 0;
}
