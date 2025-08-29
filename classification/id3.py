import sys,json,math,collections

def entropy(rows,label):
    c=collections.Counter(r[label] for r in rows)
    n=len(rows); return -sum((v/n)*math.log2(v/n) for v in c.values())

def gain(rows,attr,label):
    n=len(rows); base=entropy(rows,label)
    parts=collections.defaultdict(list)
    for r in rows: parts[r[attr]].append(r)
    rem=sum((len(p)/n)*entropy(p,label) for p in parts.values())
    return base-rem

def majority(rows,label):
    return collections.Counter(r[label] for r in rows).most_common(1)[0][0]

def id3(rows,attrs,label):
    labs=[r[label] for r in rows]
    if len(set(labs))==1: return labs[0]
    if not attrs: return majority(rows,label)
    best=max(attrs,key=lambda a:gain(rows,a,label))
    tree={best:{}}
    vals=set(r[best] for r in rows)
    for v in vals:
        subset=[r for r in rows if r[best]==v]
        tree[best][v]=id3(subset,[a for a in attrs if a!=best],label)
    return tree

def predict(tree,row,default=None):
    while isinstance(tree,dict):
        a=list(tree.keys())[0]
        v=row.get(a)
        tree=tree[a].get(v,default)
        if tree is None: return default
    return tree

def rules(tree,prefix=None):
    if not isinstance(tree,dict):
        yield (" AND ".join(prefix) if prefix else "TRUE", tree); return
    a=list(tree.keys())[0]
    for v,sub in tree[a].items():
        yield from rules(sub,(prefix or [])+[f"{a}={v}"])

if __name__=="__main__":
    path=sys.argv[1]
    label=sys.argv[2] if len(sys.argv)>2 else "label"
    data=json.load(open(path))
    attrs=[k for k in data[0].keys() if k!=label]
    tree=id3(data,attrs,label)
    print("TREE:",json.dumps(tree,indent=2))
    for cond,y in rules(tree):
        print(f"IF {cond} THEN {label}={y}")
