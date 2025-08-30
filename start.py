import sys, json
from models.pattern.apriori import apriori

if __name__ == "__main__":
    file, minsup, minconf = sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    with open(file) as f: tx = json.load(f)
    freq, rules = apriori([set(t) for t in tx], minsup, minconf)
    print(freq, rules)
