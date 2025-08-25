import json
from itertools import combinations
from collections import defaultdict

def apriori(transactions, min_support=0.5, min_conf=0.7):
    n=len(transactions)
    # L1
    counts=defaultdict(int)
    for t in transactions:
        for i in t: counts[frozenset([i])]+=1
    L={i:c/n for i,c in counts.items() if c/n>=min_support}
    freq=[L]; allf=dict(L); k=2
    while L:
        C=[a|b for a in L for b in L if len(a|b)==k]
        C=[c for c in set(C) if all(frozenset(s) in L for s in combinations(c,k-1))]
        counts=defaultdict(int)
        for t in transactions:
            for c in C:
                if c.issubset(t): counts[c]+=1
        L={c:v/n for c,v in counts.items() if v/n>=min_support}
        allf.update(L)
        if L: freq.append(L)
        k+=1
    rules=[]
    for iset,sup in allf.items():
        if len(iset)<2: continue
        for r in range(1,len(iset)):
            for X in combinations(iset,r):
                X=frozenset(X); Y=iset-X
                if allf[X]>0:
                    conf=sup/allf[X]
                    if conf>=min_conf: rules.append((set(X),set(Y),sup,conf))
    return freq,rules

if __name__=="__main__":
    with open("db/tx.json") as f: 
        tx=json.load(f)
    freq,rules=apriori([set(t) for t in tx],0.5,0.7)
    print("Frequent:",freq)
    print("Rules:",rules)
