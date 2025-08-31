import json, sys, math, collections

# Train NB on categorical features with a "count" weight per row
def train_tabular(rows, target, features, alpha=1.0):
    # priors
    cls_counts = collections.Counter()
    N = 0
    for r in rows:
        w = int(r.get("count", 1))
        cls_counts[r[target]] += w
        N += w
    logpri = {y: math.log(cls_counts[y] / N) for y in cls_counts}

    # vocab per feature and conditional counts
    vocab = {f:set() for f in features}
    cond = {f:{y:collections.Counter() for y in cls_counts} for f in features}
    totals = {f:{y:0 for y in cls_counts} for f in features}

    for r in rows:
        y = r[target]; w = int(r.get("count", 1))
        for f in features:
            v = r[f]
            vocab[f].add(v)
            cond[f][y][v] += w
            totals[f][y] += w

    # log P(f=v | y) with Laplace smoothing per feature
    loglik = {f:{y:{} for y in cls_counts} for f in features}
    unk = {}  # store log-prob for unseen values per (f,y)
    for f in features:
        V = max(len(vocab[f]), 1)
        for y in cls_counts:
            denom = totals[f][y] + alpha * V
            loglik[f][y] = {v: math.log((cond[f][y][v] + alpha) / denom) for v in vocab[f]}
            unk[(f,y)] = math.log(alpha / denom)

    return {"type":"tabular_nb","target":target,"features":features,
            "logpri":logpri,"loglik":loglik,"vocab":{f:list(v) for f in vocab},
            "unk":{f+"|"+str(y):v for (f,y),v in unk.items()}}

def predict(model, x):
    logpri,loglik,features = model["logpri"], model["loglik"], model["features"]
    def unk_log(f,y): return model["unk"][f+"|"+str(y)]
    scores = {}
    for y in logpri:
        s = logpri[y]
        for f in features:
            v = x.get(f, None)
            s += loglik[f][y].get(v, unk_log(f,y))
        scores[y] = s
    yhat = max(scores, key=scores.get)
    # return normalized probs too
    m = max(scores.values())
    probs = {y: math.exp(scores[y]-m) for y in scores}
    Z = sum(probs.values()) or 1.0
    probs = {y: probs[y]/Z for y in probs}
    return yhat, probs

if __name__=="__main__":
    # Usage:
    # python models/classification/naive_bayes_tabular.py db/raw/hr.json department status,age,salary '{"status":"junior","age":"26..30","salary":"26K..30K"}'
    path = sys.argv[1]; target = sys.argv[2]; features = sys.argv[3].split(",")
    rows = json.load(open(path))
    model = train_tabular(rows, target, features, alpha=1.0)
    if len(sys.argv) > 4:
        q = json.loads(sys.argv[4])
        yhat, probs = predict(model, q)
        print("Predict:", yhat)
        print("Probs:", probs)
    else:
        print("Model trained. Classes:", list(model["logpri"].keys()))
