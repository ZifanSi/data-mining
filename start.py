import sys, json, time
from models.pattern.apriori import apriori
from models.pattern.fp_growth import fpgrowth
from models.pattern.hmine import hmine_mine

if __name__=="__main__":
    algo, file = sys.argv[1], sys.argv[2]
    with open(file) as f: tx = json.load(f)

    start = time.time()

    if algo=="apriori":
        minsup, minconf = float(sys.argv[3]), float(sys.argv[4])
        freq, rules = apriori([set(t) for t in tx], minsup, minconf)
        print(freq, rules)
    elif algo=="fp":
        minsup = float(sys.argv[3])
        print(fpgrowth(tx, minsup))
    elif algo=="hmine":
        minsup = float(sys.argv[3])
        print(hmine_mine(tx, minsup))
    else:
        print("unknown algo")

    print(f"Runtime: {time.time()-start:.4f} seconds")
