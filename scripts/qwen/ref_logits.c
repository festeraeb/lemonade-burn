// ref_logits <model.gguf> <mode> <ids.txt> <out>
//  mode "logits": ids.txt has one prompt per line (space-separated token ids, no BOS). For every position writes the full
//                 f32 logits (n_vocab floats) to <out>, positions of all prompts concatenated; teacher forcing.
//  mode "greedy": ids.txt as above; for each prompt prints the prompt ids and 48 greedily generated ids (one line each) to <out>.
// Written for the engine-123 correctness gate; uses llama.cpp (MIT) as the reference implementation.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llama.h"
static int readline_ids(FILE*f,int**ids){char*line=NULL;size_t cap=0;ssize_t n=getline(&line,&cap,f);if(n<=0)return -1;int c=0,m=1024;int*a=malloc(m*4);char*p=line,*e;while(1){long v=strtol(p,&e,10);if(e==p)break;if(c==m){m*=2;a=realloc(a,m*4);}a[c++]=v;p=e;}free(line);*ids=a;return c;}
int main(int argc,char**argv){
  llama_backend_init();
  struct llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=0;
  struct llama_model*model=llama_model_load_from_file(argv[1],mp); if(!model){fprintf(stderr,"load failed\n");return 1;}
  const struct llama_vocab*vocab=llama_model_get_vocab(model); int nv=llama_vocab_n_tokens(vocab);
  struct llama_context_params cp=llama_context_default_params(); cp.n_ctx=2048; cp.n_batch=1024; cp.n_ubatch=512; cp.n_threads=atoi(getenv("NT")?getenv("NT"):"16"); cp.n_threads_batch=cp.n_threads;
  struct llama_context*ctx=llama_init_from_model(model,cp); if(!ctx){fprintf(stderr,"ctx failed\n");return 1;}
  if(!strcmp(argv[2],"tok")||!strcmp(argv[2],"detok")){ // tok: text lines (\\n = newline) -> ids lines ; detok: ids lines -> text (json-ish escaped)
    FILE*in=fopen(argv[3],"r"),*out=fopen(argv[4],"w"); char*line=NULL;size_t cap=0;ssize_t k;
    while((k=getline(&line,&cap,in))>0){ if(line[k-1]=='\n')line[--k]=0;
      if(argv[2][0]=='t'){ char*t=malloc(k+1);int m=0;for(int i=0;i<k;i++){if(line[i]=='\\'&&line[i+1]=='n'){t[m++]='\n';i++;}else t[m++]=line[i];}
        int*ids=malloc(4*(m+16));int c=llama_tokenize(vocab,t,m,ids,m+16,false,true);for(int i=0;i<c;i++)fprintf(out,"%d ",ids[i]);fprintf(out,"\n");free(t);free(ids);}
      else{ char*p=line;char*e;while(1){long v=strtol(p,&e,10);if(e==p)break;char buf[256];int w=llama_token_to_piece(vocab,v,buf,255,0,true);for(int i=0;i<w;i++){if(buf[i]=='\n')fputs("\\n",out);else fputc(buf[i],out);}p=e;}fprintf(out,"\n");}
    } return 0;}
  int greedy=!strcmp(argv[2],"greedy"); int one=!strcmp(argv[2],"logits1"); FILE*in=fopen(argv[3],"r"); FILE*out=fopen(argv[4],"wb");
  int*ids; int n; struct llama_batch b=llama_batch_init(1024,0,1);
  while((n=readline_ids(in,&ids))>0){
    llama_memory_clear(llama_get_memory(ctx),true);
    if(one){for(int i=0;i<n;i++){b.n_tokens=1;b.token[0]=ids[i];b.pos[0]=i;b.n_seq_id[0]=1;b.seq_id[0][0]=0;b.logits[0]=1;if(llama_decode(ctx,b)){fprintf(stderr,"decode failed\n");return 2;}fwrite(llama_get_logits_ith(ctx,0),4,nv,out);}free(ids);continue;}
    b.n_tokens=n; for(int i=0;i<n;i++){b.token[i]=ids[i];b.pos[i]=i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=greedy?(i==n-1):1;}
    if(llama_decode(ctx,b)){fprintf(stderr,"decode failed\n");return 2;}
    if(!greedy){for(int i=0;i<n;i++)fwrite(llama_get_logits_ith(ctx,i),4,nv,out);}
    else{
      for(int i=0;i<n;i++)fprintf(out,"%d ",ids[i]); fprintf(out,"|");
      int pos=n; float*lg=llama_get_logits_ith(ctx,n-1);
      for(int s=0;s<48;s++){int best=0;for(int v=1;v<nv;v++)if(lg[v]>lg[best])best=v; fprintf(out," %d",best);
        b.n_tokens=1;b.token[0]=best;b.pos[0]=pos++;b.n_seq_id[0]=1;b.seq_id[0][0]=0;b.logits[0]=1;
        if(llama_decode(ctx,b)){fprintf(stderr,"decode failed\n");return 2;} lg=llama_get_logits_ith(ctx,0);}
      fprintf(out,"\n");
    }
    free(ids);
  }
  fclose(out); return 0;}
