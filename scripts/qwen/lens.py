import json,sys,time,urllib.request
port=sys.argv[1]
base="The survey vessel crossed the sandbank at slack water while the magnetometer logged a steady baseline and the crew noted every change. "
def req(n):
    text=(base*40)[:n*6]
    b={"model":"lfm","messages":[{"role":"user","content":"Summarise in one sentence: "+text}],"max_tokens":8,"stream":True,"stream_options":{"include_usage":True},"temperature":0}
    t0=time.time(); tf=None; u=None
    with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",json.dumps(b).encode(),{"content-type":"application/json"}),timeout=300) as r:
        for raw in r:
            l=raw.decode().strip()
            if not l.startswith("data:") or l[5:].strip()=="[DONE]": continue
            o=json.loads(l[5:]); u=o.get("usage") or u
            for c in o.get("choices") or []:
                if (c.get("delta") or {}).get("content") and tf is None: tf=time.time()-t0
    return round(tf or -1,2),(u or {}).get("prompt_tokens")
for n in [60,60,300,300,420,420,200,700,700,50]:
    print(n,req(n))
