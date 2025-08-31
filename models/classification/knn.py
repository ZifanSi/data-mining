import json, sys, math, collections

def _dist(a, b, p=2):
    return sum(abs(ai-bi)**p for ai,bi in zip(a,b))**(1.0/p)

def _vote(neigh):
    cnt=collections.Counter()
    for d,y in neigh:
        cnt[y]+=1
    m=max(cnt.values())
    tied=[c for c,v in cnt.items() if v==m]
    if len(tied)==1: return tied[0]
    best=tied[0]; best_avg=1e18
    for c in tied:
        ds=[d for d,y in neigh if y==c]
        avg=sum(ds)/len(ds)
        if avg<best_avg: best, best_avg=c, avg
    return best

def knn_classify(train, q, k=3, p=2, weighted=False):
    dists=[]
    for r in train:
        d=_dist(r["x"], q, p)
        w=1.0/max(d,1e-12) if weighted else 1.0
        dists.append((d, r["y"], w))
    dists.sort(key=lambda t:t[0])
    neigh=dists[:k]
    if not weighted:
        return _vote([(d,y) for d,y,_ in neigh])
    scores=collections.Counter()
    for d,y,w in neigh: scores[y]+=w
    m=max(scores.values())
    tied=[c for c,v in scores.items() if v==m]
    if len(tied)==1: return tied[0]
    tied.sort(key=lambda c: min(d for d,y,_ in neigh if y==c))
    return tied[0]

def knn_regress(train, q, k=3, p=2, weighted=False):
    dists=[]
    for r in train:
        d=_dist(r["x"], q, p)
        w=1.0/max(d,1e-12) if weighted else 1.0
        dists.append((d, r["y"], w))
    dists.sort(key=lambda t:t[0])
    neigh=dists[:k]
    if not weighted:
        return sum(y for _,y,_ in neigh)/len(neigh)
    sw=sum(w for _,_,w in neigh)
    return sum(y*w for _,y,w in neigh)/max(sw,1e-12)

if __name__=="__main__":
    path=sys.argv[1]; mode=sys.argv[2]
    k=int(sys.argv[3]) if len(sys.argv)>3 else 3
    p=int(sys.argv[4]) if len(sys.argv)>4 else 2

    weighted=False; q=None
    if len(sys.argv)>5:
        s=sys.argv[5].lower()
        if s in ("w","weighted"):
            weighted=True
            q=json.loads(sys.argv[6]) if len(sys.argv)>6 else None
        else:
            q=json.loads(sys.argv[5])

    data=json.load(open(path))
    if mode=="cls":
        if q is None: raise SystemExit("need query vector")
        print("Predict:", knn_classify(data, q, k, p, weighted))
    elif mode=="reg":
        if q is None: raise SystemExit("need query vector")
        print("Predict:", knn_regress(data, q, k, p, weighted))
    else:
        print("mode must be cls|reg")
