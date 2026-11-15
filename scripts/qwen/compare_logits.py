import numpy as np,sys,json
V=248320
a=np.memmap(sys.argv[1],dtype=np.float32,mode='r'); b=np.memmap(sys.argv[2],dtype=np.float32,mode='r')
n=a.size//V; ag=0; kls=[]; md=[]; marg=[]
for i in range(n):
    x=a[i*V:(i+1)*V].astype(np.float64); y=b[i*V:(i+1)*V].astype(np.float64)
    lx=x-x.max(); lx=lx-np.log(np.exp(lx).sum()); ly=y-y.max(); ly=ly-np.log(np.exp(ly).sum())
    kls.append(float((np.exp(lx)*(lx-ly)).sum())); md.append(float(np.abs(x-y).max())); ag+=int(x.argmax()==y.argmax())
print(n,'positions: top1 agree',ag,round(100*ag/n,2),'% meanKL',round(float(np.mean(kls)),5),'max logit diff max',round(max(md),3),'mean',round(float(np.mean(md)),3))
