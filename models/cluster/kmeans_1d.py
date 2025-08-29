import json,sys

def assign(xs,cs):
    cls=[[] for _ in cs]
    for x in xs:
        j=min(range(len(cs)),key=lambda k:abs(x-cs[k]))
        cls[j].append(x)
    return cls

def recompute(cls,cs):
    return [sum(c)/len(c) if c else cs[j] for j,c in enumerate(cls)]

def kmeans(xs,k,it=100):
    xs=sorted(xs)
    step=max(1,len(xs)//k)
    cs=[xs[min(i*step,len(xs)-1)] for i in range(k)]
    for _ in range(it):
        cls=assign(xs,cs)
        new=recompute(cls,cs)
        if all(abs(a-b)<1e-9 for a,b in zip(cs,new)): break
        cs=new
    return cs,cls

if __name__=="__main__":
    path=sys.argv[1]; k=int(sys.argv[2])
    xs=json.load(open(path))
    cs,cls=kmeans(xs,k)
    print("Centers:",cs)
    print("Clusters:",cls)
