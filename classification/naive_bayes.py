import sys, json, collections

def train(rows, label="salary"):
    priors=collections.Counter(); conds={}; totals=collections.Counter(); vocab=collections.defaultdict(set)
    for r in rows:
        c=r[label]; n=r["count"]; priors[c]+=n
        conds.setdefault(c, collections.Counter())
        for k,v in r.items():
            if k in (label,"count"): continue
            conds[c][(k,v)]+=n; totals[(c,k)]+=n; vocab[k].add(v)
    V={k:len(vs) for k,vs in vocab.items()}
    N=sum(priors.values())
    return priors, conds, totals, V, N

def predict(query, priors, conds, totals, V, N, label="salary"):
    scores={}
    for c in priors:
        p=priors[c]/N
        for k,v in query.items():
            num=conds[c].get((k,v),0)+1
            den=totals[(c,k)]+V[k]
            p*=num/den
        scores[c]=p
    s=sum(scores.values())
    if s: 
        for c in scores: scores[c]/=s
    return scores

if __name__=="__main__":
    if len(sys.argv)<3:
        print("Usage: python Classification/naive_bayes.py db/t_employee.json department=systems status=junior age=26..30")
        sys.exit(1)

    data=json.load(open(sys.argv[1],encoding="utf-8"))
    priors,conds,totals,V,N=train(data,"salary")

    # parse query only from CLI args
    query={kv.split("=",1)[0]:kv.split("=",1)[1] for kv in sys.argv[2:]}

    res=predict(query,priors,conds,totals,V,N,"salary")
    print("Query:",query)
    for c,p in sorted(res.items(), key=lambda x:-x[1]):
        print(f"{c}: {p:.4f}")
