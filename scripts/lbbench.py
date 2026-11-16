import json,sys,time,urllib.request
port=int(sys.argv[1]); mode=sys.argv[2]; n=int(sys.argv[3]) if len(sys.argv)>3 else 128
def call(prompt,max_tokens,temp=0,stream=True):
    body=json.dumps({"model":"test","messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":temp,"stream":True}).encode()
    req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",body,{"Content-Type":"application/json"})
    t0=time.time(); first=None; ntok=0; text=""
    with urllib.request.urlopen(req,timeout=900) as r:
        for line in r:
            line=line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            j=json.loads(line[5:])
            d=j["choices"][0].get("delta",{}).get("content")
            if d:
                if first is None: first=time.time()
                ntok+=1; text+=d
    t1=time.time()
    return dict(ttft=(first or t1)-t0,tokens=ntok,dec_tps=(ntok-1)/(t1-first) if first and ntok>1 else 0,total=t1-t0,text=text)
code="Write a Rust function that parses a CSV line into fields, handling quoted fields with embedded commas and escaped quotes. Include unit tests."
if mode=="decode":
    for i in range(3):
        r=call(code,n); print(json.dumps({k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items() if k!="text"}))
elif mode=="long":
    # ~2k tokens of Rust source-like text
    src=open("/proc/self/exe","rb") if False else None
    filler="fn add(a: i32, b: i32) -> i32 { a + b } // helper number {i}\n"
    big="".join(filler.replace("{i}",str(i)).replace("add","add%d"%i) for i in range(int(sys.argv[4]) if len(sys.argv)>4 else 180))
    import random; p="Seed %d. Here is some code:\n"%random.randint(0,10**9)+big+"\nSummarize in one sentence what these functions do."
    for i in range(2):
        r=call(p if i==0 else p.replace("Seed","Seed ",1)+"x",n); print(json.dumps({k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items() if k!="text"}))
elif mode=="text":
    r=call(code,n); print(r["text"])
