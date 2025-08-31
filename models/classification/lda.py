import sys,json,statistics

def lda(train):
    classes=set(r["y"] for r in train)
    means={c:[statistics.mean([r["x"][i] for r in train if r["y"]==c]) for i in range(len(train[0]["x"]))] for c in classes}
    priors={c:sum(1 for r in train if r["y"]==c)/len(train) for c in classes}
    return means,priors

def predict(x,means,priors):
    # simple linear discriminant: argmax (mean·x + log prior)
    scores={c:sum(m[i]*x[i] for i in range(len(x)))+math.log(priors[c]) for c,m in means.items()}
    return max(scores,key=scores.get)

if __name__=="__main__":
    import math
    data=json.load(open(sys.argv[1]))
    q=json.loads(sys.argv[2])
    means,priors=lda(data)
    print("Predict:",predict(q,means,priors))
