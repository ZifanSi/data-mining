import sys, json, math, collections, random

def entropy(rows, label):
    c = collections.Counter(r[label] for r in rows)
    n = len(rows)
    return -sum((v/n)*math.log2(v/n) for v in c.values() if v>0)

def gain_ratio(rows, attr, label):
    n = len(rows); base = entropy(rows,label)
    parts=collections.defaultdict(list)
    for r in rows: parts[r[attr]].append(r)
    info = sum((len(p)/n)*entropy(p,label) for p in parts.values())
    gain = base-info
    split_info = -sum((len(p)/n)*math.log2(len(p)/n) for p in parts.values() if len(p)>0)
    return gain/split_info if split_info>0 else 0

def majority(rows,label):
    return collections.Counter(r[label] for r in rows).most_common(1)[0][0]

def c45(rows,attrs,label):
    labs=[r[label] for r in rows]
    if len(set(labs))==1: return labs[0]
    if not attrs: return majority(rows,label)
    best=max(attrs,key=lambda a: gain_ratio(rows,a,label))
    tree={best:{}}
    for v in set(r[best] for r in rows):
        subset=[r for r in rows if r[best]==v]
        tree[best][v]=c45(subset,[a for a in attrs if a!=best],label) if subset else majority(rows,label)
    return tree

def predict(tree,row,default=None):
    while isinstance(tree,dict):
        a=list(tree.keys())[0]; v=row.get(a)
        tree=tree[a].get(v,default)
        if tree is None: return default
    return tree

def train_boost(data,attrs,label,rounds=3):
    n=len(data); weights=[1/n]*n; models=[]
    for _ in range(rounds):
        # sample according to weights
        sample=random.choices(data,weights=weights,k=n)
        tree=c45(sample,attrs,label)
        preds=[predict(tree,r) for r in data]
        err=sum(w for (r,w,p) in zip(data,weights,preds) if p!=r[label])
        if err==0 or err>=0.5: continue
        alpha=0.5*math.log((1-err)/err)
        models.append((tree,alpha))
        # update weights
        new_weights=[]
        Z=sum(weights[i]*math.exp(-alpha if preds[i]==data[i][label] else alpha) for i in range(n))
        for i in range(n):
            w=weights[i]*math.exp(-alpha if preds[i]==data[i][label] else alpha)
            new_weights.append(w/Z)
        weights=new_weights
    return models

def boosted_predict(models,row,default=None):
    scores=collections.Counter()
    for tree,alpha in models:
        y=predict(tree,row,default)
        if y is not None: scores[y]+=alpha
    return scores.most_common(1)[0][0] if scores else default

if __name__=="__main__":
    path=sys.argv[1]; label=sys.argv[2] if len(sys.argv)>2 else "label"
    data=json.load(open(path))
    attrs=[k for k in data[0].keys() if k!=label]
    models=train_boost(data,attrs,label,rounds=5)
    for i,m in enumerate(models,1):
        print(f"Tree {i}: weight={m[1]:.3f}")
    # test prediction on first row
    print("Example row:",data[0])
    print("Predicted:",boosted_predict(models,data[0]))
