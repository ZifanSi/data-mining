import json, sys, os, random

CHOICES = [
    ["a"],
    ["b"],
    ["a", "b"],
    []
]

if __name__ == "__main__":
    # args: number_of_transactions
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    tx = [random.choice(CHOICES) for _ in range(n)]

    outfile = "db/raw/tx.json"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    with open(outfile, "w") as f:
        json.dump(tx, f, indent=2)

    print(f"Wrote {n} random a/b transactions -> {outfile}")
