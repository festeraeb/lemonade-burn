import sys; sys.path.insert(0,'/home/cesarops/src/llama.cpp-mainline/gguf-py')
import gguf, collections, json
r=gguf.GGUFReader(sys.argv[1])
meta={}
for k,f in r.fields.items():
    if k.startswith('tokenizer.ggml.tokens') or k.startswith('tokenizer.ggml.merges') or k.startswith('tokenizer.ggml.token_type') or k=='tokenizer.chat_template': 
        meta[k]='<%s len %d>'%(f.types[-1].name, len(f.data)); continue
    try: meta[k]=f.contents()
    except Exception as e: meta[k]='?'
json.dump(meta,open(sys.argv[2],'w'),indent=1,default=str)
c=collections.OrderedDict()
for t in r.tensors:
    key=(t.tensor_type.name)
    c.setdefault(key,[0,0,[]]); c[key][0]+=1; c[key][1]+=int(t.n_bytes); 
    c[key][2].append(t.name)
for k,v in c.items(): print(k,v[0],'tensors',round(v[1]/1e9,3),'GB')
json.dump([(t.name,t.tensor_type.name,list(map(int,t.shape)),int(t.n_bytes),int(t.data_offset)) for t in r.tensors],open(sys.argv[2]+'.tensors','w'))
