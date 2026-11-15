import json,sys,time,urllib.request,statistics
port=sys.argv[1]; model=sys.argv[2]
P=["Write a long, detailed essay about the history of sonar and how side-scan sonar is used to find shipwrecks.","Write a Rust program that reads a CSV file with quoted fields, groups rows by the first column and prints per-group totals, with explanations.","Explain in detail the causes and consequences of the 1913 Great Lakes storm.","Describe step by step how a magnetometer survey is planned, run and interpreted to find a steel wreck.","Explain what a mixture-of-experts language model is, how routing works, and why it is cheaper to run than a dense model."]
res=[]
for p in P:
    b={"model":model,"messages":[{"role":"user","content":p}],"max_tokens":256,"stream":True,"stream_options":{"include_usage":True},"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
    t0=time.time(); tf=None; usage=None; txt=[]
    with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",json.dumps(b).encode(),{"content-type":"application/json"}),timeout=900) as r:
        for raw in r:
            l=raw.decode().strip()
            if not l.startswith("data:") or l[5:].strip()=="[DONE]": continue
            o=json.loads(l[5:])
            if o.get("usage"): usage=o["usage"]
            for c in o.get("choices") or []:
                d=c.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    if tf is None: tf=time.time()-t0
                    txt.append(d.get("content") or "")
    tot=time.time()-t0; n=(usage or {}).get("completion_tokens",0)
    res.append((round((n-1)/(tot-tf),1),n,round(tf,2),"".join(txt)[:60].replace("\n"," ")))
print(json.dumps({"tps":[r[0] for r in res],"n":[r[1] for r in res],"first_s":[r[2] for r in res],"median_all":statistics.median(r[0] for r in res),"median_2_5":statistics.median(r[0] for r in res[1:])}))
for r in res: print("  ",r)
