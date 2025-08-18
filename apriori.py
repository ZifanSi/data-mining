from __future__ import annotations
import argparse
from collections import defaultdict
from itertools import combinations
from typing import List, Set, Tuple, Dict, FrozenSet
import math

def parse_args():
    p = argparse.ArgumentParser(description="Apriori frequent itemset mining")
    p.add_argument("data", help="Path to transactions file (CSV or whitespace-separated)")
    p.add_argument("--minsup", type=float, required=True,
                   help="Minimum support: fraction (0,1] or absolute count (>=1)")
    p.add_argument("--minconf", type=float, default=0.6, help="Minimum confidence in [0,1]")
    p.add_argument("--delimiter", choices=[",","ws"], default=",",
                   help="Delimiter: ',' (default) or 'ws' for arbitrary whitespace")
    return p.parse_args()

def read_transactions(path: str, delimiter: str) -> List[FrozenSet[str]]:
    txns = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in (line.split(",") if delimiter == "," else line.split())]
            if not parts:
                continue
            txns.append(frozenset(sorted(parts)))
    return txns

def support_threshold(minsup: float, n_txn: int) -> int:
    if minsup <= 0:
        raise ValueError("minsup must be > 0")
    if minsup <= 1.0:
        return max(1, int(math.ceil(minsup * n_txn)))
    return int(minsup)

def count_L1(transactions: List[FrozenSet[str]]) -> Dict[FrozenSet[str], int]:
    counts = defaultdict(int)
    for t in transactions:
        for item in t:
            counts[frozenset([item])] += 1
    return counts

def make_Lk(counts: Dict[FrozenSet[str], int], minsup_count: int) -> List[FrozenSet[str]]:
    return [itemset for itemset, c in counts.items() if c >= minsup_count]

def join_step(Lk: List[FrozenSet[str]]) -> List[FrozenSet[str]]:
    Lk_sorted = sorted([tuple(sorted(s)) for s in Lk])
    k = len(Lk_sorted[0]) if Lk_sorted else 0
    Ck1 = set()
    for i in range(len(Lk_sorted)):
        for j in range(i+1, len(Lk_sorted)):
            if Lk_sorted[i][:k-1] == Lk_sorted[j][:k-1]:
                c = tuple(sorted(set(Lk_sorted[i]) | set(Lk_sorted[j])))
                if len(c) == k+1:
                    Ck1.add(frozenset(c))
            else:
                break
    return sorted(Ck1, key=lambda s: tuple(sorted(s)))

def prune_step(Ck1: List[FrozenSet[str]], Lk: Set[FrozenSet[str]]) -> List[FrozenSet[str]]:
    if not Ck1:
        return []
    k = len(next(iter(Ck1))) - 1
    pruned = []
    for c in Ck1:
        if all(frozenset(s) in Lk for s in combinations(c, k)):
            pruned.append(c)
    return pruned

def count_support(Ck: List[FrozenSet[str]], transactions: List[FrozenSet[str]]) -> Dict[FrozenSet[str], int]:
    counts = defaultdict(int)
    for t in transactions:
        for c in Ck:
            if c.issubset(t):
                counts[c] += 1
    return counts

def apriori(transactions: List[FrozenSet[str]], minsup_count: int):
    support_counts: Dict[FrozenSet[str], int] = defaultdict(int)
    L1_counts = count_L1(transactions)
    Lk = make_Lk(L1_counts, minsup_count)
    support_counts.update({s: L1_counts[s] for s in Lk})

    all_frequents = [sorted(Lk, key=lambda s: tuple(sorted(s)))]
    while Lk:
        Ck1 = prune_step(join_step(Lk), set(Lk))
        if not Ck1:
            break
        Ck1_counts = count_support(Ck1, transactions)
        Lk1 = make_Lk(Ck1_counts, minsup_count)
        support_counts.update({s: Ck1_counts[s] for s in Lk1})
        if not Lk1:
            break
        all_frequents.append(sorted(Lk1, key=lambda s: tuple(sorted(s))))
        Lk = Lk1
    return all_frequents, support_counts

def generate_rules(all_frequents, support_counts, n_txn: int, minconf: float):
    rules = []
    for level in all_frequents[1:]:
        for F in level:
            items = list(F)
            for r in range(1, len(items)):
                for X in combinations(items, r):
                    X = frozenset(X)
                    Y = F - X
                    suppF = support_counts[F] / n_txn
                    conf = support_counts[F] / support_counts[X]
                    suppY = support_counts[Y] / n_txn if Y in support_counts else 0
                    lift = conf / suppY if suppY > 0 else float("inf")
                    if conf >= minconf:
                        rules.append((X, Y, suppF, conf, lift))
    rules.sort(key=lambda t: (-t[3], -t[2], tuple(sorted(t[0]))))
    return rules

def main():
    args = parse_args()
    txns = read_transactions(args.data, args.delimiter)
    minsup_count = support_threshold(args.minsup, len(txns))
    all_frequents, support_counts = apriori(txns, minsup_count)

    print(f"# Transactions: {len(txns)}")
    print(f"# Minimum support count: {minsup_count}")
    print("\n== Frequent itemsets ==")
    for level_idx, level in enumerate(all_frequents, start=1):
        if not level: continue
        print(f"\n--- size {level_idx} ---")
        for s in level:
            supp = support_counts[s] / len(txns)
            print(f"{'{' + ', '.join(sorted(s)) + '}'}  supp={supp:.3f} (count={support_counts[s]})")

    print("\n== Association rules ==")
    rules = generate_rules(all_frequents, support_counts, len(txns), args.minconf)
    if not rules:
        print("(none met minconf)")
    else:
        for X, Y, suppF, conf, lift in rules:
            xs = "{" + ", ".join(sorted(X)) + "}"
            ys = "{" + ", ".join(sorted(Y)) + "}"
            print(f"{xs} -> {ys}  conf={conf:.3f}  lift={lift:.3f}  supp(F)={suppF:.3f}")

if __name__ == "__main__":
    main()
