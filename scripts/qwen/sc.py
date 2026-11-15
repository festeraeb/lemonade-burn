import json,sys,time,urllib.request,os
# sc.py out.json  : streamed requests like seat_demand.chat: hard prompt x3, short prompt x2
sysm=open(os.path.dirname(os.path.abspath(__file__))+"/hard-system.txt").read(); hp=open(os.path.dirname(os.path.abspath(__file__))+"/hard-prompt.txt").read()
cases=[("hard",[{"role":"system","content":sysm},{"role":"user","content":hp}]) for _ in range(3)]+[("short",[{"role":"user","content":"Explain what a mixture-of-experts language model is and why it is cheap to run."}]) for _ in range(2)]
N=int(os.environ.get("NTOK","300")); res=[]
for name,m in cases:
    b={"model":"lfm","messages":m,"max_tokens":N,"stream":True,"stream_options":{"include_usage":True},"temperature":float(os.environ.get("TEMP","0.2"))}
    if os.environ.get("NOTHINK"): b["chat_template_kwargs"]={"enable_thinking":False}
    t0=time.time(); tf=None; ch=0; usage=None
    with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8198/v1/chat/completions",json.dumps(b).encode(),{"content-type":"application/json"}),timeout=900) as r:
        for raw in r:
            l=raw.decode().strip()
            if not l.startswith("data:") or l[5:].strip()=="[DONE]": continue
            o=json.loads(l[5:])
            if o.get("usage"): usage=o["usage"]
            for c in o.get("choices") or []:
                d=c.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    if tf is None: tf=time.time()-t0
                    ch+=1
    tot=time.time()-t0; n=(usage or {}).get("completion_tokens") or ch
    res.append({"case":name,"prompt_tokens":(usage or {}).get("prompt_tokens"),"first_s":round(tf,2),"n":n,"tps_after_first":round((n-1)/(tot-tf),2)})
json.dump(res,open(sys.argv[1],"w"))
print(" | ".join(f'{r["case"]} p={r["prompt_tokens"]} first={r["first_s"]}s {r["tps_after_first"]}t/s' for r in res))
