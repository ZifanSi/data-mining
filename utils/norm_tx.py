import json, sys, os
from collections import Counter
from itertools import combinations

def summarize(transactions, max_k=2, lowercase=True, sort_items=True, drop_empty=True):
    # normalize transactions
    norm = []
    for t in transactions:
        if drop_empty and not t: 
            continue
        items = [i.strip() for i in t if isinstance(i, str)]
        if lowercase:
            items = [i.lower() for i in items]
        if sort_items:
            items.sort()
        if items:
            norm.append(items)

    n = len(norm)
    counts = Counter()

    for t in norm:
        for k in range(1, max_k+1):
            for comb in combinations(t, k):
                counts["".join(comb)] += 1

    return {"transactions": n, "counts": dict(counts)}

if __name__ == "__main__":
    infile  = sys.argv[1] if len(sys.argv) > 1 else "db/raw/tx.json"
    base    = os.path.splitext(os.path.basename(infile))[0]
    outfile = f"db/processed/{base}.json"

    with open(infile) as f:
        tx = json.load(f)

    summary = summarize(tx)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary -> {outfile}")
