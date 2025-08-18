import sys, math
from collections import Counter

def read_txns(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip().split(",") for line in f if line.strip()]

def minsup_count(minsup, n):
    return int(math.ceil(minsup*n)) if 0 < minsup <= 1 else int(minsup)

def prepare(txns, minsup_cnt):
    cnt = Counter(i for t in txns for i in set(t))
    f_items = [i for i, c in cnt.items() if c >= minsup_cnt]
    f_list  = sorted(f_items, key=lambda x: (-cnt[x], x))
    pos = {i: k for k, i in enumerate(f_list)}
    prepped = []
    for t in txns:
        items = [i for i in t if i in pos]
        if items:
            items.sort(key=lambda i: pos[i])
            prepped.append(items)
    return prepped, f_list

def hmine(txns, f_list, minsup_cnt, prefix, results, n):
    if not f_list or not txns: return
    cnt = Counter()
    for t in txns:
        cnt.update(set(t))

    for a in reversed(f_list):
        supp = cnt[a]
        if supp < minsup_cnt: 
            continue
        newp = prefix + [a]
        results.append((newp, supp / n))

        proj = []
        for t in txns:
            if a in t:
                ai = t.index(a)
                suf = t[ai+1:]
                if suf: proj.append(suf)

        if not proj: 
            continue
        local_cnt = Counter(i for u in proj for i in set(u))
        local_f = [i for i in f_list if i != a and local_cnt[i] >= minsup_cnt]
        if local_f:
            hmine(proj, local_f, minsup_cnt, newp, results, n)

if __name__ == "__main__":
    path = sys.argv[1]; minsup = float(sys.argv[2])
    txns = read_txns(path); n = len(txns); m = minsup_count(minsup, n)
    prepped, f_list = prepare(txns, m)
    results = []
    hmine(prepped, f_list, m, [], results, n)
    for items, supp in sorted(results, key=lambda x: (len(x[0]), x[0])):
        print(f"{items} supp={supp:.2f}")
