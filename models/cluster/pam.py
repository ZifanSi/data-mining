import json, sys, random, math

def _v(x): return x if isinstance(x, list) else [x]
def _d(a,b): 
    a,b=_v(a),_v(b)
    return math.sqrt(sum((ai-bi)**2 for ai,bi in zip(a,b)))

def _assign(X, med_idx):
    meds=[X[i] for i in med_idx]
    labels=[]; cost=0.0
    for x in X:
        j=min(range(len(meds)), key=lambda t:_d(x, meds[t]))
        labels.append(j); cost+=_d(x, meds[j])
    return labels, cost

def pam(X, k, it=100):
    n=len(X)
    if k<=0 or k>n: raise ValueError("bad k")
    # init medoids by spread-out indices
    idx=list(range(n)); step=max(1,n//k)
    med_idx=sorted({min(i*step, n-1) for i in range(k)})
    labels, best_cost=_assign(X, med_idx)

    for _ in range(it):
        improved=False
        for mi in list(med_idx):
            for h in idx:
                if h in med_idx: continue
                cand=sorted((med_idx - {mi}) | {h}) if isinstance(med_idx, set) else sorted([x for x in med_idx if x!=mi]+[h])
                _, cost=_assign(X, cand)
                if cost+1e-12 < best_cost:
                    med_idx=cand; best_cost=cost; improved=True
        if not improved: break
    # build clusters
    labels,_=_assign(X, med_idx)
    meds=[X[i] for i in med_idx]
    clusters=[[] for _ in meds]
    for x,l in zip(X, labels): clusters[l].append(x)
    return meds, clusters, best_cost

if __name__=="__main__":
    path=sys.argv[1]; k=int(sys.argv[2])
    X=json.load(open(path))
    meds,cls,cost=pam(X,k)
    print("Medoids:", meds)
    print("Clusters:", cls)
    print("Total cost:", cost)
