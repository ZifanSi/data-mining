import json, sys, math
from collections import Counter

def minsup_count(minsup,n): return int(math.ceil(minsup*n)) if 0<minsup<=1 else int(minsup)

def prepare(txns,m):
    cnt=Counter(i for t in txns for i in set(t))
    f_items=[i for i,c in cnt.items() if c>=m]
    f_list=sorted(f_items,key=lambda x:(-cnt[x],x))
    pos={i:k for k,i in enumerate(f_list)}
    prepped=[]
    for t in txns:
        items=[i for i in t if i in pos]
        if items:
            items.sort(key=lambda i:pos[i])
            prepped.append(items)
    return prepped,f_list

def hmine(txns,f_list,m,prefix,res,n):
    if not f_list or not txns: return
    cnt=Counter()
    for t in txns: cnt.update(set(t))
    for a in reversed(f_list):
        supp=cnt.get(a,0)
        if supp<m: continue
        newp=prefix+[a]; res.append((newp,supp/n))
        proj=[t[t.index(a)+1:] for t in txns if a in t and t.index(a)+1<len(t)]
        if not proj: continue
        local_cnt=Counter(i for u in proj for i in set(u))
        local_f=[i for i in f_list if i!=a and local_cnt.get(i,0)>=m]
        if local_f: hmine(proj,local_f,m,newp,res,n)

def hmine_mine(transactions, minsup):
    n=len(transactions); m=minsup_count(minsup,n)
    prepped,f_list=prepare(transactions,m); res=[]
    hmine(prepped,f_list,m,[],res,n)
    return res

if __name__=="__main__":
    path=sys.argv[1]; minsup=float(sys.argv[2]) if len(sys.argv)>2 else 0.5
    with open(path) as f: tx=json.load(f)
    print("Frequent:", hmine_mine(tx, minsup))
