import json, sys

def assign(xs, cs):
    cls=[[] for _ in cs]
    for x in xs:
        j=min(range(len(cs)), key=lambda k: abs(x-cs[k]))
        cls[j].append(x)
    return cls

def median(c):
    if not c: return None
    c=sorted(c); n=len(c); m=n//2
    return c[m] if n%2 else 0.5*(c[m-1]+c[m])

def recompute(cls, cs):
    out=[]
    for j,c in enumerate(cls):
        m=median(c)
        out.append(cs[j] if m is None else m)
    return out

def kmedians(xs, k, it=100):
    xs=sorted(xs)
    step=max(1, len(xs)//k)
    cs=[xs[min(i*step, len(xs)-1)] for i in range(k)]  # simple deterministic init
    for _ in range(it):
        cls=assign(xs, cs)
        new=recompute(cls, cs)
        if all(abs(a-b)<1e-9 for a,b in zip(cs,new)): break
        cs=new
    return cs, cls

if __name__=="__main__":
    path=sys.argv[1]; k=int(sys.argv[2])
    xs=json.load(open(path))
    cs,cls=kmedians(xs,k)
    print("Centers:", cs)
    print("Clusters:", cls)
