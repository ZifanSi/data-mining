import json, sys, random, math

def _as_vec(x): return x if isinstance(x, list) else [x]
def _dist2(a,b):
    a,b=_as_vec(a),_as_vec(b)
    return sum((ai-bi)**2 for ai,bi in zip(a,b))

def fcm(X,k,m=2.0,it=100,eps=1e-8):
    n=len(X); d=len(_as_vec(X[0]))
    C=random.sample(X,k) if n>=k else [X[0]]*k  # init centers
    U=[[0.0]*k for _ in range(n)]
    for _ in range(it):
        # E-step: memberships
        for i,x in enumerate(X):
            djs=[max(_dist2(x,c),eps) for c in C]
            for j in range(k):
                denom=sum((djs[j]/djq)**(1.0/(m-1)) for djq in djs)
                U[i][j]=1.0/denom
        # M-step: centers
        C_new=[]
        changed=False
        for j in range(k):
            w=[(U[i][j]**m) for i in range(n)]
            sW=sum(w) or eps
            c=[0.0]*d
            for i in range(n):
                xi=_as_vec(X[i])
                for t in range(d): c[t]+=w[i]*xi[t]
            c=[ct/sW for ct in c]; C_new.append(c)
            if any(abs(c[t]-C[j][t])>1e-9 for t in range(d)): changed=True
        C=C_new
        if not changed: break
    return C,U

if __name__=="__main__":
    path=sys.argv[1]; k=int(sys.argv[2])
    m=float(sys.argv[3]) if len(sys.argv)>3 else 2.0
    X=json.load(open(path))
    C,U=fcm(X,k,m)
    print("Centers:",C)
    print("Membership (first 10 rows):", [u for u in U[:10]])
