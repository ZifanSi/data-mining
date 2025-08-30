import json, sys, os, random

if __name__ == "__main__":
    # args: length max_value
    length = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    maxval = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    raw = [random.randint(0, maxval) for _ in range(length)]

    outfile = "db/raw/freq.json"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    with open(outfile, "w") as f:
        json.dump(raw, f, indent=2)

    print(f"Wrote raw data -> {outfile}")
