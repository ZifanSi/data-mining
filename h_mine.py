import sys, math
from collections import Counter, defaultdict

def read_txns(path):
    with open(path) as f:
        return [line.strip().split(",") for line in f if line.strip()]

def minsup_count(minsup, n):
    return int(math.ceil(minsup*n)) if 0 < minsup <= 1 else int(minsup)

def hmine(txns, minsup, prefix, results, n):
    counts=Counter(i for t in txns for i in set(t))
    items=[i for i,c in counts.items() if c>=minsup]
    items.sort(key=lambda i:(-counts[i],i))
    for a in items:
        supp=sum(1 for t in txns if a in t)
        if supp>=minsup:
            newp=prefix+[a]
            results.append((newp,supp/n))
            # build projected db
            proj=[ [i for i in t[t.index(a)+1:] if i in items] for t in txns if a in t ]
            proj=[p for p in proj if p]
            if proj: hmine(proj,minsup,newp,results,n)

if __name__=="__main__":
    path=sys.argv[1]; minsup=float(sys.argv[2])
    txns=read_txns(path); n=len(txns); m=minsup_count(minsup,n)
    results=[]; hmine(txns,m,[],results,n)
    for items,supp in sorted(results,key=lambda x:(len(x[0]),x[0])):
        print(f"{items} supp={supp:.2f}")
