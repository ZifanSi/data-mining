import json, sys, math, random

def dot(a,b): return sum(x*y for x,y in zip(a,b))
def add(a,b): return [x+y for x,y in zip(a,b)]
def sub(a,b): return [x-y for x,y in zip(a,b)]
def scal(a,s): return [s*x for x in a]
def norm(a): return math.sqrt(dot(a,a))
def mean_col(X):  # column means
    n=len(X); d=len(X[0])
    m=[0.0]*d
    for x in X:
        for j in range(d): m[j]+=x[j]
    return [v/n for v in m]

def center(X):
    mu=mean_col(X)
    return [sub(x,mu) for x in X], mu

def cov_mat(Xc):  # (1/n) X^T X
    n=len(Xc); d=len(Xc[0])
    C=[[0.0]*d for _ in range(d)]
    for x in Xc:
        for i in range(d):
            xi=x[i]
            for j in range(i,d):
                C[i][j]+=xi*x[j]
    for i in range(d):
        for j in range(i,d):
            C[i][j]/=n; C[j][i]=C[i][j]
    return C

def matvec(M,v): return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]
def proj(u,v):  return scal(u, dot(u,v)/max(dot(u,u),1e-12))
def orthonormalize(v, basis):
    w=v[:]
    for u in basis: w=sub(w, proj(u,w))
    n=norm(w) or 1.0
    return [w_i/n for w_i in w]

def topk_eig(C,k,it=200):
    d=len(C); vecs=[]
    for _ in range(min(k,d)):
        v=[random.random() for _ in range(d)]
        v=orthonormalize(v, vecs)
        for _ in range(it):
            v_new=matvec(C,v)
            v_new=orthonormalize(v_new, vecs)
            if norm(sub(v_new,v))<1e-9: break
            v=v_new
        vecs.append(v)
    return vecs  # columns (each length d)

def transform(Xc, comps):  # project onto components
    # Y = Xc * W  (n x d) @ (d x k)
    n=len(Xc); k=len(comps)
    # transpose comps -> W[d][k]
    W=[[comps[j][i] for j in range(k)] for i in range(len(comps[0]))]
    Y=[[sum(Xc[i][t]*W[t][j] for t in range(len(W))) for j in range(k)] for i in range(n)]
    return Y

def pca(X,k):
    Xc,mu=center(X)
    C=cov_mat(Xc)
    comps=topk_eig(C,k)
    Y=transform(Xc, comps)
    return mu, comps, Y

if __name__=="__main__":
    path=sys.argv[1]; k=int(sys.argv[2])
    X=json.load(open(path))
    mu,comps,Y=pca(X,k)
    print("Mean:", mu)
    print("Components (each is a PC):", comps)
    print("Projected (first 10 rows):", Y[:10])
