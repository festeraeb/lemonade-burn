import json,urllib.request,sys
base="The survey vessel crossed the sandbank at slack water while the magnetometer logged a steady baseline and the crew noted every change. "
out=[]
for n in (300,700):
    b={"model":"lfm","messages":[{"role":"user","content":"Summarise in two sentences: "+(base*40)[:n*6]}],"max_tokens":40,"temperature":0}
    r=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8196/v1/chat/completions",json.dumps(b).encode(),{"content-type":"application/json"}),timeout=300))
    out.append(r["choices"][0]["message"]["content"])
print(json.dumps(out))
