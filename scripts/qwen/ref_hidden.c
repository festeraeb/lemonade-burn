// ref_hidden <model.gguf> <ids-line.txt> <out.bin> : decode the first prompt line in one batch and dump the f32 tensors named
// attn_residual / linear_attn_out / attn_output / ffn_out / post_moe per layer (all tokens), via the eval callback.
// Record: name (NUL-terminated, "name-layer"), n_tokens (i32), n_embd (i32), floats [n_tokens][n_embd].
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llama.h"
#include "ggml-backend.h"
static FILE*out;
static int want(const char*n){const char*k[]={"attn_residual-","linear_attn_out-","attn_output-","ffn_out-","post_moe-",0};for(int i=0;k[i];i++)if(!strncmp(n,k[i],strlen(k[i])))return 1;return 0;}
static bool cb(struct ggml_tensor*t,bool ask,void*ud){
  if(ask)return want(t->name)&&t->type==GGML_TYPE_F32;
  if(!want(t->name)||t->type!=GGML_TYPE_F32)return true;
  int ne0=t->ne[0],ne1=t->ne[1]; size_t nb=(size_t)ne0*ne1*4; float*buf=malloc(nb); ggml_backend_tensor_get(t,buf,0,nb);
  fwrite(t->name,1,strlen(t->name)+1,out); fwrite(&ne1,4,1,out); fwrite(&ne0,4,1,out); fwrite(buf,1,nb,out); free(buf); return true;}
int main(int argc,char**argv){
  llama_backend_init(); struct llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=0;
  struct llama_model*model=llama_model_load_from_file(argv[1],mp);
  struct llama_context_params cp=llama_context_default_params(); cp.n_ctx=512; cp.n_batch=512; cp.n_ubatch=512; cp.n_threads=16; cp.n_threads_batch=16; cp.cb_eval=cb; cp.cb_eval_user_data=NULL;
  struct llama_context*ctx=llama_init_from_model(model,cp);
  FILE*in=fopen(argv[2],"r"); out=fopen(argv[3],"wb"); int ids[512],n=0; while(fscanf(in,"%d",&ids[n])==1&&n<511)n++;
  struct llama_batch b=llama_batch_init(512,0,1); b.n_tokens=n; for(int i=0;i<n;i++){b.token[i]=ids[i];b.pos[i]=i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=(i==n-1);}
  if(llama_decode(ctx,b)){fprintf(stderr,"decode failed\n");return 2;}
  fclose(out);return 0;}
