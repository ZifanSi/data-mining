import json, sys
from collections import defaultdict

def vertical_format(transactions):
    V = defaultdict(set)
    for tid, t in enumerate(transactions):
        for item in set(t):
            V[frozenset([item])].add(tid)
    return V

def eclat(transactions, minsup):
    n = len(transactions)
    m = int(minsup * n) if 0 < minsup <= 1 else int(minsup)
    V = vertical_format(transactions)
    res = []

    def dfs(P, T, tail):
        if len(T) >= m:
            res.append((sorted(P), len(T)/n))
        for i, (Q, U) in enumerate(tail):
            IU = T & U
            if len(IU) >= m:
                dfs(P | Q, IU, tail[i+1:])

    items = sorted(V.items(), key=lambda kv: (-len(kv[1]), next(iter(kv[0]))))
    for i, (Q, T) in enumerate(items):
        if len(T) >= m:
            dfs(Q, T, items[i+1:])

    return res

if __name__ == "__main__":
    path, minsup = sys.argv[1], float(sys.argv[2])
    with open(path) as f: tx = json.load(f)
    print(eclat(tx, minsup))
