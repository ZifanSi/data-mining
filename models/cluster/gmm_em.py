import json, sys, math, random

def _as_vec(x): return x if isinstance(x,list) else [x]
def _dist2(a,b):
    a,b=_as_vec(a),_as_vec(b)
    return sum((ai-bi)**2 for ai,bi in zip(a,b))

def _gauss_iso(x,mu,var):
    d=len(_as_vec(x)); var=max(var,1e-8)
    c=(2*math.pi*var)**(-d/2)
    return c*math.exp(-0.5*_dist2(x,mu)/var)

def gmm_em(X,k,it=100,eps=1e-8):
    n=len(X); d=len(_as_vec(X[0]))
    mus=random.sample(X,k) if n>=k else [X[0]]*k
    pis=[1.0/k]*k
    # init var as overall variance
    mean=[sum(_as_vec(x)[t] for x in X)/n for t in range(d)]
    var=sum(_dist2(x,mean) for x in X)/(n*d) if n>0 else 1.0
    vars=[var]*k

    for _ in range(it):
        # E-step
        R=[[0.0]*k for _ in range(n)]
        for i,x in enumerate(X):
            num=[pis[j]*_gauss_iso(x,mus[j],vars[j]) for j in range(k)]
            s=sum(num) or eps
            for j in range(k): R[i][j]=num[j]/s
        # M-step
        Nk=[sum(R[i][j] for i in range(n)) for j in range(k)]
        for j in range(k):
            Nj=max(Nk[j],eps)
            # mean
            mu=[0.0]*d
            for i,x in enumerate(X):
                xv=_as_vec(x)
                for t in range(d): mu[t]+=R[i][j]*xv[t]
            mus[j]=[v/Nj for v in mu]
            # variance (isotropic)
            s=0.0
            for i,x in enumerate(X): s+=R[i][j]*_dist2(x,mus[j])
            vars[j]=max(s/(Nj*d),1e-8)
            pis[j]=Nj/n
    return mus,vars,pis

if __name__=="__main__":
    path=sys.argv[1]; k=int(sys.argv[2])
    X=json.load(open(path))
    mus,vars,pis=gmm_em(X,k)
    print("Means:",mus)
    print("Vars:",vars)
    print("Priors:",pis)
