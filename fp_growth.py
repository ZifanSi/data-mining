import sys, math
from collections import Counter, defaultdict

def read_txns(path):
    with open(path) as f:
        return [line.strip().split(",") for line in f if line.strip()]

def minsup_count(minsup, n):
    return int(math.ceil(minsup*n)) if 0 < minsup <= 1 else int(minsup)

class Node:
    def __init__(self, item, parent):
        self.item, self.count, self.parent = item, 1, parent
        self.children, self.next = {}, None

def build_tree(txns, minsup):
    counts = Counter(i for t in txns for i in set(t))
    items = [i for i,c in counts.items() if c >= minsup]
    order = {i: idx for idx,i in enumerate(sorted(items, key=lambda i:(-counts[i], i)))}
    header = {i: None for i in items}
    root = Node(None,None)

    for t in txns:
        filt = [i for i in t if i in items]
        filt.sort(key=lambda i: order[i])
        node = root
        for i in filt:
            if i not in node.children:
                child = Node(i,node); node.children[i]=child
                # link header
                if header[i] is None: header[i]=child
                else:
                    cur=header[i]
                    while cur.next: cur=cur.next
                    cur.next=child
            else: node.children[i].count+=1
            node=node.children[i]
    return root, header, order

def ascend(node):
    path=[]
    while node.parent and node.parent.item:
        node=node.parent; path.append(node.item)
    return list(reversed(path))

def mine(tree, header, minsup, prefix, results, n):
    for item,node in header.items():
        supp=0; cond_db=[]
        cur=node
        while cur:
            supp+=cur.count
            for _ in range(cur.count):
                cond_db.append(ascend(cur))
            cur=cur.next
        if supp>=minsup:
            newp=prefix+[item]
            results.append((newp,supp/n))
            if cond_db:
                r,h,_=build_tree(cond_db,minsup)
                if h: mine(r,h,minsup,newp,results,n)

if __name__=="__main__":
    path=sys.argv[1]; minsup=float(sys.argv[2])
    txns=read_txns(path); n=len(txns); m=minsup_count(minsup,n)
    r,h,o=build_tree(txns,m); results=[]
    mine(r,h,m,[],results,n)
    for items,supp in sorted(results,key=lambda x:(len(x[0]),x[0])):
        print(f"{items} supp={supp:.2f}")
