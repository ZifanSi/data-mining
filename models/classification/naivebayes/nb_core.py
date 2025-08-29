import json, math
from collections import Counter, defaultdict

class NaiveBayes:
    def __init__(self,label="salary",alpha=1.0,prior_mode="empirical"):
        self.label=label; self.alpha=float(alpha); self.prior_mode=prior_mode
        self.priors=Counter(); self.conds=defaultdict(Counter); self.totals=Counter()
        self.V={}; self.N=0; self.attrs=[]

    def fit_count_table(self,b):
        self.label=b["label"]; self.priors=b["priors"]; self.conds=b["conds"]
        self.totals=b["totals"]; self.V=b["V"]; self.N=b["N"]; self.attrs=b["attrs"]
        if self.prior_mode=="uniform":
            k=len(self.priors); total=self.N
            self.priors=Counter({c: total/k for c in self.priors})

    def _p_cat(self,c,a,v):
        num=self.conds[c].get((a,v),0)+self.alpha
        den=self.totals[(c,a)]+self.alpha*max(1,self.V.get(a,0))
        return num/den if den>0 else 1.0

    def predict_proba(self,q):
        scores={}; total_prior=sum(self.priors.values()) or 1.0
        for c in self.priors:
            p=(self.priors[c]/total_prior)
            for a in self.attrs:
                if a in q: p*=self._p_cat(c,a,q[a])
            scores[c]=p
        z=sum(scores.values()) or 1.0
        for c in scores: scores[c]/=z
        return scores

    def predict(self,q): 
        s=self.predict_proba(q); return max(s,key=s.get)

    def to_dict(self):
        return {
            "label":self.label,"alpha":self.alpha,"prior_mode":self.prior_mode,
            "priors":dict(self.priors),
            "conds":{c:{f"{k[0]}=={k[1]}":v for k,v in self.conds[c].items()} for c in self.conds},
            "totals":{"|".join([c,a]):v for (c,a),v in self.totals.items()},
            "V":self.V,"N":self.N,"attrs":self.attrs
        }

    @staticmethod
    def from_dict(d):
        nb=NaiveBayes(d["label"],d.get("alpha",1.0),d.get("prior_mode","empirical"))
        nb.priors=Counter(d["priors"]); nb.conds=defaultdict(Counter)
        for c,mp in d["conds"].items():
            for sk,v in mp.items():
                a,val=sk.split("==",1); nb.conds[c][(a,val)]=v
        nb.totals=Counter()
        for sk,v in d["totals"].items():
            c,a=sk.split("|",1); nb.totals[(c,a)]=v
        nb.V=d["V"]; nb.N=d["N"]; nb.attrs=d["attrs"]
        return nb

    def save(self,path): json.dump(self.to_dict(),open(path,"w",encoding="utf-8"))
    @staticmethod
    def load(path): return NaiveBayes.from_dict(json.load(open(path,encoding="utf-8")))
