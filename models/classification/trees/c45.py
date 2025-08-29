import sys, json, math, collections

def entropy(rows, label):
    c = collections.Counter(r[label] for r in rows)
    n = len(rows)
    return -sum((v/n)*math.log2(v/n) for v in c.values() if v>0)

def gain_ratio(rows, attr, label):
    n = len(rows)
    base = entropy(rows, label)
    parts = collections.defaultdict(list)
    for r in rows:
        parts[r[attr]].append(r)
    info = sum((len(p)/n)*entropy(p,label) for p in parts.values())
    gain = base - info
    split_info = -sum((len(p)/n)*math.log2(len(p)/n) for p in parts.values() if len(p)>0)
    return gain/split_info if split_info>0 else 0

def majority(rows,label):
    return collections.Counter(r[label] for r in rows).most_common(1)[0][0]

def c45(rows, attrs, label):
    labs = [r[label] for r in rows]
    if len(set(labs))==1: return labs[0]
    if not attrs: return majority(rows,label)
    best = max(attrs, key=lambda a: gain_ratio(rows,a,label))
    tree = {best:{}}
    vals = set(r[best] for r in rows)
    for v in vals:
        subset = [r for r in rows if r[best]==v]
        if not subset: tree[best][v]=majority(rows,label)
        else: tree[best][v]=c45(subset,[a for a in attrs if a!=best],label)
    return tree

def rules(tree, prefix=None):
    if not isinstance(tree, dict):
        yield (" AND ".join(prefix) if prefix else "TRUE", tree); return
    a = list(tree.keys())[0]
    for v,sub in tree[a].items():
        yield from rules(sub,(prefix or [])+[f"{a}={v}"])

if __name__=="__main__":
    path = sys.argv[1]; label = sys.argv[2] if len(sys.argv)>2 else "label"
    data = json.load(open(path))
    attrs = [k for k in data[0].keys() if k!=label]
    tree = c45(data, attrs, label)
    print("TREE:", json.dumps(tree, indent=2))
    for cond,y in rules(tree):
        print(f"IF {cond} THEN {label}={y}")
