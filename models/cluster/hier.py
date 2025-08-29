#!/usr/bin/env python3
import sys, json, math

# ---------- I/O ----------
def read_points(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            txt = f.read().strip()
            if not txt:
                raise ValueError("empty file")
            data = json.loads(txt)
            if isinstance(data, dict) and "points" in data:
                data = data["points"]
            return [to_vec(p) for p in data]
    except (json.JSONDecodeError, ValueError):
        pts=[]
        with open(path,"r",encoding="utf-8-sig") as f:
            for line in f:
                s=line.strip()
                if not s or s.startswith("#"): continue
                vals=[float(x) for x in s.split(",") if x.strip()]
                pts.append(vals)
        return pts

def to_vec(p):
    if isinstance(p,(int,float)): return [float(p)]
    return [float(x) for x in p]

# ---------- basic math ----------
def dist(a,b): return math.sqrt(sum((ai-bi)**2 for ai,bi in zip(a,b)))
def mean(points):
    d=len(points[0]); s=[0.0]*d
    for p in points:
        for j in range(d): s[j]+=p[j]
    return [v/len(points) for v in s]
def sse(points, ctr=None):
    if not points: return 0.0
    c=ctr if ctr else mean(points)
    return sum(dist(p,c)**2 for p in points)

# ---------- AGNES ----------
def inter_cluster_dist(A,B,link):
    if link=="single":   return min(dist(a,b) for a in A for b in B)
    if link=="complete": return max(dist(a,b) for a in A for b in B)
    if link=="average":
        return sum(dist(a,b) for a in A for b in B)/(len(A)*len(B))
    if link=="centroid": return dist(mean(A),mean(B))

def agnes(points,k=2,link="average"):
    clusters=[[p] for p in points]
    while len(clusters)>k:
        best=(float("inf"),None,None)
        for i in range(len(clusters)):
            for j in range(i+1,len(clusters)):
                d=inter_cluster_dist(clusters[i],clusters[j],link)
                if d<best[0]: best=(d,i,j)
        _,i,j=best
        clusters[i]+=clusters[j]; del clusters[j]
    cents=[mean(c) for c in clusters]
    labels=[min(range(len(cents)),key=lambda t:dist(p,cents[t])) for p in points]
    return labels,clusters

# ---------- DIANA ----------
def two_means(points,iters=10):
    c1,c2=points[0],points[-1]
    for _ in range(iters):
        A,B=[],[]
        for p in points:
            (A if dist(p,c1)<=dist(p,c2) else B).append(p)
        if not A or not B: break
        nc1,nc2=mean(A),mean(B)
        if all(abs(a-b)<1e-12 for a,b in zip(c1,nc1)) and all(abs(a-b)<1e-12 for a,b in zip(c2,nc2)): break
        c1,c2=nc1,nc2
    return (A if A else points, B if B else [])

def diana(points,k=2):
    clusters=[points[:]]
    while len(clusters)<k:
        idx=max(range(len(clusters)),key=lambda i:sse(clusters[i]))
        A,B=two_means(clusters[idx])
        if not B: break
        clusters[idx]=A; clusters.append(B)
    cents=[mean(c) for c in clusters]
    labels=[min(range(len(cents)),key=lambda j:dist(p,cents[j])) for p in points]
    return labels,clusters

# ---------- BIRCH-lite ----------
class CF:
    __slots__=("n","ls","ss")
    def __init__(self,d): self.n=0; self.ls=[0.0]*d; self.ss=[0.0]*d
    def add(self,x):
        self.n+=1
        for j,v in enumerate(x): self.ls[j]+=v; self.ss[j]+=v*v
    def centroid(self): return [v/self.n for v in self.ls]
    def radius(self):
        mu=self.centroid()
        norm_ss=sum(self.ss)
        mu_dot_ls=sum(mu[j]*self.ls[j] for j in range(len(mu)))
        mu_norm=sum(m*m for m in mu)
        avg_sq=(norm_ss-2*mu_dot_ls+self.n*mu_norm)/self.n
        return math.sqrt(max(0.0,avg_sq))

def birch(points,radius=1.0):
    d=len(points[0]); clusters=[]
    for p in points:
        best_i,best_r=None,None
        for i,(cf,mem) in enumerate(clusters):
            tmp=CF(d); tmp.n, tmp.ls[:], tmp.ss[:]=cf.n, cf.ls[:], cf.ss[:]
            tmp.add(p); r=tmp.radius()
            if r<=radius and (best_r is None or r<best_r):
                best_i,best_r=i,r
        if best_i is None:
            cf=CF(d); cf.add(p); clusters.append((cf,[p]))
        else:
            cf,mem=clusters[best_i]; cf.add(p); mem.append(p)
    cents=[cf.centroid() for cf,_ in clusters]
    labels=[min(range(len(cents)),key=lambda j:dist(p,cents[j])) for p in points]
    return labels,[mem for _,mem in clusters]

# ---------- CLI ----------
def get_arg(argv,name,default=None,req=False,cast=str):
    if name in argv:
        i=argv.index(name)
        if i+1>=len(argv): sys.exit("missing value for "+name)
        return cast(argv[i+1])
    if req: sys.exit("required: "+name)
    return default

def main(argv):
    if len(argv)<3:
        print("Usage:\n  python cluster/hier.py agnes <file> --k K [--link single|complete|average|centroid]"
              "\n  python cluster/hier.py diana <file> --k K"
              "\n  python cluster/hier.py birch <file> --radius R"); sys.exit(1)
    algo=argv[1].lower(); path=argv[2]
    pts=read_points(path)
    if algo=="agnes":
        k=get_arg(argv,"--k",req=True,cast=int)
        link=get_arg(argv,"--link",default="average")
        lbl,clu=agnes(pts,k,link)
    elif algo=="diana":
        k=get_arg(argv,"--k",req=True,cast=int)
        lbl,clu=diana(pts,k)
    elif algo=="birch":
        R=get_arg(argv,"--radius",req=True,cast=float)
        lbl,clu=birch(pts,R)
    else: sys.exit("unknown algo: "+algo)
    print("labels:",lbl)
    print("clusters:",[[tuple(p) for p in c] for c in clu])

if __name__=="__main__":
    main(sys.argv)
