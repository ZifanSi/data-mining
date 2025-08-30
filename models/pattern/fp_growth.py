import json, sys, math
from collections import Counter

class Node:
    def __init__(s, i, p): s.item, s.count, s.parent = i, 1, p; s.children, s.next = {}, None

def build_tree(txns, m):
    c = Counter(i for t in txns for i in set(t))
    it = [i for i,v in c.items() if v >= m]
    o = {i:j for j,i in enumerate(sorted(it, key=lambda x: (-c[x], x)))}
    h = {i: None for i in it}; r = Node(None, None)
    for t in txns:
        f = sorted([i for i in t if i in it], key=lambda x: o[x]); n = r
        for i in f:
            if i not in n.children:
                ch = Node(i, n); n.children[i] = ch
                if not h[i]: h[i] = ch
                else:
                    cur = h[i]
                    while cur.next: cur = cur.next
                    cur.next = ch
            else: n.children[i].count += 1
            n = n.children[i]
    return r, h

def ascend(n):
    p = []
    while n.parent and n.parent.item:
        n = n.parent; p.append(n.item)
    return p[::-1]

def mine(h, m, p, res, n):
    for i, node in h.items():
        s, db = 0, []
        cur = node
        while cur:
            s += cur.count; db += [ascend(cur)] * cur.count; cur = cur.next
        if s >= m:
            newp = p + [i]; res.append((newp, s / n))
            if db:
                r, h2 = build_tree(db, m)
                if h2: mine(h2, m, newp, res, n)

def fpgrowth(txns, minsup):
    n = len(txns); m = int(math.ceil(minsup * n))
    _, h = build_tree([set(t) for t in txns], m)
    res = []; mine(h, m, [], res, n)
    return res

if __name__ == "__main__":
    path = sys.argv[1]; minsup = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    with open(path) as f: tx = json.load(f)
    print("Frequent:", fpgrowth(tx, minsup))
