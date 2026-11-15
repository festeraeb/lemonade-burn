// ggml_dequant <gguf> <tensors.tsv> <out.hash>: for every tensor and every row, dequantise with llama.cpp's own
// to_float (ggml type traits) and write FNV-1a64 of the f32 bit patterns, one u64 per row, in tsv order.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include "ggml.h"
static uint64_t fnv(const uint8_t*p,size_t n){uint64_t h=1469598103934665603ULL;for(size_t i=0;i<n;i++){h^=p[i];h*=1099511628211ULL;}return h;}
int main(int argc,char**argv){
  int fd=open(argv[1],O_RDONLY); off_t sz=lseek(fd,0,SEEK_END);
  uint8_t*m=mmap(0,sz,PROT_READ,MAP_SHARED,fd,0);
  FILE*tf=fopen(argv[2],"r"); FILE*out=fopen(argv[3],"wb"); char name[512]; long type,ncols,nrows; unsigned long long off;
  float*buf=malloc(1<<20);
  while(fscanf(tf,"%511s %ld %ld %llu %ld",name,&type,&ncols,&off,&nrows)==5){
    size_t rb=ggml_row_size((enum ggml_type)type,ncols);
    const struct ggml_type_traits*tr=ggml_get_type_traits((enum ggml_type)type);
    if(ncols*4>(1<<20)){fprintf(stderr,"wide\n");return 3;}
    for(long r=0;r<nrows;r++){
      const uint8_t*src=m+off+r*rb;
      if(type==0)memcpy(buf,src,ncols*4); else tr->to_float(src,buf,ncols);
      uint64_t h=fnv((uint8_t*)buf,ncols*4); fwrite(&h,8,1,out);
    }
  }
  return 0;}
