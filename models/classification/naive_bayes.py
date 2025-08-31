#!/usr/bin/env python3
import json, sys, math, collections, argparse

# ---------- core ----------
def _w(r):
    try: return int(r.get("count", 1))
    except: return 0

def train_tabular(rows, target, features, alpha=1.0):
    cls_counts = collections.Counter(); N = 0
    for r in rows:
        w = _w(r)
        cls_counts[r[target]] += w; N += w
    logpri = {y: math.log(cls_counts[y] / N) for y in cls_counts}

    vocab = {f: set() for f in features}
    cond  = {f: {y: collections.Counter() for y in cls_counts} for f in features}
    totals= {f: {y: 0 for y in cls_counts} for f in features}

    for r in rows:
        y = r[target]; w = _w(r)
        for f in features:
            v = r.get(f, None)
            vocab[f].add(v)
            cond[f][y][v] += w
            totals[f][y] += w

    loglik = {f: {y: {} for y in cls_counts} for f in features}
    unk = {}
    for f in features:
        V = max(len(vocab[f]), 1)
        for y in cls_counts:
            denom = totals[f][y] + alpha * V
            loglik[f][y] = {v: math.log((cond[f][y][v] + alpha) / denom) for v in vocab[f]}
            unk[(f, y)] = math.log(alpha / denom)

    return {
        "target": target, "features": features,
        "logpri": logpri, "loglik": loglik,
        "vocab": {f: sorted(list(v)) for f, v in vocab.items()},
        "unk": {f"{f}|{y}": v for (f, y), v in unk.items()},
        "class_counts": dict(cls_counts), "N": N, "alpha": alpha
    }

def predict(model, x):
    def unk_log(f,y): return model["unk"][f"{f}|{y}"]
    scores, contribs = {}, {}
    for y in model["logpri"]:
        s = model["logpri"][y]; contribs[y] = {"<PRIOR>": s}
        for f in model["features"]:
            v = x.get(f, None)
            term = model["loglik"][f][y].get(v, unk_log(f,y))
            s += term; contribs[y][f] = term
        scores[y] = s
    yhat = max(scores, key=scores.get)
    m = max(scores.values())
    probs = {y: math.exp(scores[y]-m) for y in scores}
    Z = sum(probs.values()) or 1.0
    probs = {y: probs[y]/Z for y in probs}
    return yhat, probs, contribs

def autodetect_schema(rows):
    keys = set().union(*(r.keys() for r in rows))
    tgt = "department" if "department" in keys else None
    if not tgt:
        for k in keys:
            if k == "count": continue
            if any(isinstance(r.get(k), str) for r in rows): tgt = k; break
    tgt = tgt or next((k for k in keys if k != "count"), "label")
    feats = [k for k in keys if k not in (tgt, "count")]; feats.sort()
    return tgt, feats

# ---------- printing ----------
def pct(x): return f"{100.0*x:.2f}%"

def print_summary(rows, model):
    total_weight = sum(_w(r) for r in rows)
    print("# Dataset")
    print(f"- Groups: {len(rows)}")
    print(f"- Weighted samples: {total_weight}")
    print(f"- Target: {model['target']}")
    print(f"- Features: {', '.join(model['features'])}\n")

    print("# Class Priors")
    N = model["N"]
    for y,c in sorted(model["class_counts"].items(), key=lambda kv:(-kv[1], kv[0])):
        p = c/N if N else 0.0
        print(f"- {y}: count={c} prior={p:.6f} ({pct(p)})")
    print()

    print("# Feature Cardinality")
    for f in model["features"]:
        print(f"- {f}: |V|={len(model['vocab'][f])}")
    print()

def print_small_checks(rows, target, features):
    bad_count = sum(1 for r in rows if not isinstance(r.get("count",1), (int,float)) or _w(r) <= 0)
    keys = [target] + features
    sig = collections.Counter(tuple((k, r.get(k,None)) for k in keys) for r in rows)
    dups = sum(1 for c in sig.values() if c > 1)
    print("# Quick Checks")
    print(f"- Rows with invalid/nonpositive count: {bad_count}")
    print(f"- Duplicate groups (same {keys}): {dups}\n")

def print_prediction(model, x, yhat, probs, contribs, topk=None):
    print("# Prediction")
    print(f"- Input: {json.dumps(x)}")
    print(f"- Predicted: {yhat}\n")
    print("# Probabilities")
    items = sorted(probs.items(), key=lambda kv: -kv[1])
    if topk: items = items[:topk]
    for y,p in items:
        print(f"- {y}: {p:.6f} ({pct(p)})")
    print("\n# Explanation (log-score terms)")
    parts = contribs[yhat]
    print(f"- <PRIOR>: {parts['<PRIOR>']:.6f}")
    for f in model["features"]:
        v = x.get(f, None)
        used = "seen" if v in model["loglik"][f][yhat] else "UNK"
        print(f"- {f}={v} ({used}): {parts[f]:.6f}")

# ---------- cli ----------
def main():
    # avoid unicode issues on Windows consoles
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

    ap = argparse.ArgumentParser(description="Simple, practical NB for categorical tables with 'count' weights.")
    ap.add_argument("path")
    ap.add_argument("--target")
    ap.add_argument("--features")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--predict", help='JSON, e.g. {"status":"junior","age":"26..30","salary":"26K..30K"}')
    ap.add_argument("--topk", type=int, default=0, help="Show top-K classes in probability output (0=all)")
    args = ap.parse_args()

    rows = json.load(open(args.path, "r", encoding="utf-8"))
    target, features = (args.target, None)
    if not target or not args.features:
        auto_t, auto_f = autodetect_schema(rows)
        target = target or auto_t
        features = auto_f if not args.features else [s for s in args.features.split(",") if s]
    else:
        features = [s for s in args.features.split(",") if s]
    if target in features: features = [f for f in features if f != target]

    model = train_tabular(rows, target, features, alpha=args.alpha)
    print_summary(rows, model)
    print_small_checks(rows, target, features)

    if args.predict:
        x = json.loads(args.predict)
        yhat, probs, contribs = predict(model, x)
        print_prediction(model, x, yhat, probs, contribs, topk=(args.topk or None))

if __name__ == "__main__":
    main()
