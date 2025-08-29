#!/usr/bin/env python3
import sys, json, argparse, os
from classification_models.naivebayes.nb_data import load_count_table
from classification_models.naivebayes.nb_core import NaiveBayes
from classification_models.naivebayes.nb_metrics import accuracy

def parse_kv_list(kvlist):
    q={}
    for kv in kvlist:
        k,v=kv.split("=",1); q[k]=v
    return q

def main(argv=None):
    ap=argparse.ArgumentParser(description="Naive Bayes CLI")
    sub=ap.add_subparsers(dest="cmd",required=True)

    t=sub.add_parser("train")
    t.add_argument("input"); t.add_argument("--label",default="salary")
    t.add_argument("--alpha",type=float,default=1.0)
    t.add_argument("--prior",choices=["empirical","uniform"],default="empirical")
    t.add_argument("--model",default="nb_employee.json")

    p=sub.add_parser("predict")
    p.add_argument("--model",required=True)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--query",nargs="+")
    g.add_argument("--batch")

    args=ap.parse_args(argv)

    if args.cmd=="train":
        bundle=load_count_table(args.input,label=args.label)
        nb=NaiveBayes(label=args.label,alpha=args.alpha,prior_mode=args.prior)
        nb.fit_count_table(bundle)

        # always save into trained_data/
        os.makedirs("trained_data",exist_ok=True)
        model_path=os.path.join("trained_data",os.path.basename(args.model))
        nb.save(model_path)
        print("saved:",model_path)

    elif args.cmd=="predict":
        model_path=args.model
        if not os.path.isfile(model_path):
            # also check inside trained_data/
            alt=os.path.join("trained_data",os.path.basename(model_path))
            if os.path.isfile(alt): model_path=alt
        nb=NaiveBayes.load(model_path)

        if args.query:
            q=parse_kv_list(args.query)
            probs=nb.predict_proba(q); pred=max(probs,key=probs.get)
            print("query:",q)
            for c,p in sorted(probs.items(),key=lambda x:-x[1]): print(f"{c}: {p:.4f}")
            print("pred:",pred)
        else:
            batch=json.load(open(args.batch,encoding="utf-8"))
            y_true=[]; y_pred=[]
            for q in batch:
                y=q.pop(nb.label,None)
                pred=nb.predict(q); print({"query":q,"pred":pred})
                if y is not None: y_true.append(y); y_pred.append(pred)
            if y_true: print("accuracy:",f"{accuracy(y_true,y_pred):.4f}")

if __name__=="__main__":
    main()
