import json
from collections import Counter, defaultdict

def load_count_table(path, label="salary"):
    rows=json.load(open(path,encoding="utf-8"))
    priors=Counter(); conds=defaultdict(Counter); totals=Counter(); vocab=defaultdict(set)
    for r in rows:
        c=r[label]; n=int(r.get("count",1)); priors[c]+=n
        for k,v in r.items():
            if k in (label,"count"): continue
            conds[c][(k,v)]+=n; totals[(c,k)]+=n; vocab[k].add(v)
    V={k:len(vs) for k,vs in vocab.items()}
    N=sum(priors.values()); attrs=sorted(vocab.keys())
    return {"priors":priors,"conds":conds,"totals":totals,"V":V,"N":N,"attrs":attrs,"label":label}

def expand_count_table_to_instances(path, label="salary"):
    rows=json.load(open(path,encoding="utf-8")); out=[]
    for r in rows:
        n=int(r.get("count",1)); base={k:v for k,v in r.items() if k!="count"}
        out.extend([base.copy() for _ in range(n)])
    return out
