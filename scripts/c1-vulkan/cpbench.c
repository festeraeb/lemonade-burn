#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <emmintrin.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char**argv){
  int fd=open(argv[1],O_RDONLY);struct stat st;fstat(fd,&st);
  char*m=mmap(0,st.st_size,PROT_READ,MAP_SHARED,fd,0);
  madvise(m,st.st_size,MADV_POPULATE_READ);
  size_t N=1769472;int reps=300;
  char*dst=aligned_alloc(4096,N);memset(dst,1,N);
  srand(1);
  size_t span=st.st_size/N-2;
  double t0=now();
  for(int i=0;i<reps;i++){size_t o=(size_t)(rand()%span)*N;memcpy(dst,m+o,N);}
  double t1=now();
  printf("mmap->anon memcpy: %.2f GB/s\n",reps*(double)N/(t1-t0)/1e9);
  // warm repeat same chunk
  t0=now();for(int i=0;i<reps;i++){memcpy(dst,m+(size_t)3*N,N);}t1=now();
  printf("same chunk (cache-warm): %.2f GB/s\n",reps*(double)N/(t1-t0)/1e9);
  // NT stores to anon
  t0=now();
  for(int i=0;i<reps;i++){size_t o=(size_t)(rand()%span)*N;
    for(size_t j=0;j<N;j+=64){__m128i a=_mm_loadu_si128((void*)(m+o+j)),b=_mm_loadu_si128((void*)(m+o+j+16)),c=_mm_loadu_si128((void*)(m+o+j+32)),d=_mm_loadu_si128((void*)(m+o+j+48));
      _mm_stream_si128((void*)(dst+j),a);_mm_stream_si128((void*)(dst+j+16),b);_mm_stream_si128((void*)(dst+j+32),c);_mm_stream_si128((void*)(dst+j+48),d);}}
  t1=now();printf("mmap->anon NT: %.2f GB/s\n",reps*(double)N/(t1-t0)/1e9);
  // read only sum
  t0=now();unsigned long s=0;
  for(int i=0;i<reps;i++){size_t o=(size_t)(rand()%span)*N;for(size_t j=0;j<N;j+=64)s+=m[o+j];}
  t1=now();printf("touch 1 line/64B: %.2f GB/s(eff) %lu\n",reps*(double)N/(t1-t0)/1e9,s);
  return 0;}
