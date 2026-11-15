import json,sys
for l in sys.argv[1:]:
    L=[json.loads(x) for x in open(l)][3:]
    f=lambda i: sum(sum(y[i] for y in t["layers"]) for t in L)/len(L)/1000
    nz=[y for t in L for y in t["layers"] if y[1]==0]
    z=lambda i: (sum(y[i] for y in nz)/len(nz)/1000) if nz else 0
    print(l.split('/')[-1],"tok_ms %.0f"%(sum(t["issue_to_last_sync_us"] for t in L)/len(L)/1000),"miss %.0f"%(sum(sum(y[1] for y in t["layers"]) for t in L)/len(L)),"| sync %.0f busy %.0f read_begin %.0f read_fin %.0f post %.0f"%(f(2),f(3),f(5),f(6),f(7)),"| zero-miss layers n=%d: sync %.2f rb %.2f rf %.2f ms"%(len(nz)/len(L),z(2),z(5),z(6)))
