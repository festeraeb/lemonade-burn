import json,sys,time,urllib.request
port=sys.argv[1]; model=sys.argv[2]
sents=["The survey vessel crossed the sandbank at slack water while the magnetometer logged a steady baseline of {a} nanotesla.","Side-scan returns {b} metres to port showed a hard linear target with a short acoustic shadow behind it.","On day {c} the second pass agreed with the first within {d} metres, which the crew noted in the margin of the log.","A buried ferrous object produces a dipole whose amplitude falls with the cube of the distance from the sensor, here about {e} metres."]
SEED=int(sys.argv[3]) if len(sys.argv)>3 else 0
text=" ".join(sents[i%4].format(a=48000+SEED+i*7,b=20+(i*13)%90,c=i+1,d=3+(i*5)%11,e=2+(i*3)%9) for i in range(115))
def req(msgs,mt=48):
    b={"model":model,"messages":msgs,"max_tokens":mt,"stream":True,"stream_options":{"include_usage":True},"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
    t0=time.time(); tf=None; usage=None; txt=[]
    with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",json.dumps(b).encode(),{"content-type":"application/json"}),timeout=900) as r:
        for raw in r:
            l=raw.decode().strip()
            if not l.startswith("data:") or l[5:].strip()=="[DONE]": continue
            o=json.loads(l[5:])
            if o.get("usage"): usage=o["usage"]
            for c in o.get("choices") or []:
                d=c.get("delta") or {}
                if d.get("content"):
                    if tf is None: tf=time.time()-t0
                    txt.append(d["content"])
    return {"ttft":round(tf or -1,2),"total":round(time.time()-t0,2),"prompt":(usage or {}).get("prompt_tokens"),"n":(usage or {}).get("completion_tokens"),"text":"".join(txt)}
u1={"role":"user","content":"Read this survey log and then answer in one sentence: what instruments are mentioned?\n\n"+text}
r1=req([u1])
r2=req([u1,{"role":"assistant","content":r1["text"].strip()},{"role":"user","content":"Now say the same in five words."}])
SEED2=int(sys.argv[3]) + 777 if len(sys.argv)>3 else 777
text2=" ".join(sents[i%4].format(a=61000+SEED2+i*11,b=30+(i*7)%80,c=i+3,d=4+(i*3)%9,e=3+(i*5)%7) for i in range(115))
r3=req([{"role":"user","content":"Read this survey log and then answer in one sentence: which direction is mentioned?\n\n"+text2}])
print(json.dumps({"fresh":{k:r3[k] for k in("ttft","total","prompt")},"turn1":{k:r1[k] for k in("ttft","total","prompt","n")},"turn2":{k:r2[k] for k in("ttft","total","prompt","n")},"a1":r1["text"][:70].replace("\n"," "),"a2":r2["text"][:50].replace("\n"," ")}))
