import json, re, math
from collections import defaultdict, Counter
from typing import Any, Dict, Iterable, List, Tuple, Optional

# ====================== basic utils ======================
def _norm_str(x: Any) -> str:
    return (str(x or "").strip())

def _norm_token(x: Any) -> str:
    return _norm_str(x).lower()

def _to_ints(s: str) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", s or "")]

def _levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    m, n = len(a), len(b)
    prev = list(range(n+1))
    curr = [0]*(n+1)
    for i in range(1, m+1):
        curr[0] = i
        ca = a[i-1]
        for j in range(1, n+1):
            cb = b[j-1]
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
        prev, curr = curr, prev
    return prev[n]

def _edit_ratio(a: str, b: str) -> float:
    a = _norm_token(a); b = _norm_token(b)
    if not a and not b: return 0.0
    L = max(len(a), len(b), 1)
    return _levenshtein(a, b) / L  # 0 identical .. 1 very different

def _letters(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch.isalpha()).lower()

def _is_subseq(short: str, long: str) -> bool:
    """Return True iff all chars of `short` appear in order in `long`."""
    it = iter(long)
    for c in short:
        for L in it:
            if L == c:
                break
        else:
            return False
    return True

# ====================== count coercion ======================
_WORD = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,
         "seven":7,"eight":8,"nine":9,"ten":10}

def coerce_count(v: Any) -> int:
    if v is None: return 1
    if isinstance(v, (int, float)): return max(1, int(v))
    s = _norm_token(v)
    if s.isdigit(): return max(1, int(s))
    return _WORD.get(s, 1)

def get_count_value(row: Dict[str,Any], count_keys: List[str]) -> int:
    for k in count_keys:
        if k in row: return coerce_count(row.get(k))
    return 1

# ====================== KNN majority for categorical ======================
class KNNMajority:
    """
    Unsupervised KNN over observed spellings (per field).
    Builds from the data itself; no labels or dictionaries.
    """
    def __init__(self, k:int=3, max_edit_ratio:float=0.45, null_policy:str="majority"):
        self.k = max(1, int(k))
        self.max_edit_ratio = float(max_edit_ratio)
        self.null_policy = null_policy  # "majority" or "unknown"
        self.vocab: List[str] = []
        self.freq: Dict[str,int] = {}
        self.global_majority: Optional[str] = None

    def fit(self, tokens_with_weights: List[Tuple[str,int]]):
        f = Counter()
        for t, w in tokens_with_weights:
            t0 = _norm_token(t)
            if not t0: continue
            f[t0] += max(1,int(w))
        self.freq = dict(f)
        self.vocab = list(self.freq.keys())
        self.global_majority = (max(self.freq.items(), key=lambda kv: kv[1])[0]
                                if self.freq else None)

    def predict(self, token: Any) -> Optional[str]:
        """
        Canonicalize even if the token itself exists in the vocab:
        - Prefer close neighbors (edit distance)
        - Recognize abbreviations (e.g., mktg -> marketing, jr -> junior)
        - Break ties by global frequency, then prefer LONGER tokens (full words)
        """
        q = _norm_token(token)
        if not q:
            return self.global_majority if self.null_policy == "majority" else "unknown"
        if not self.vocab:
            return None

        q_letters = _letters(q)
        cand = []
        for v in self.vocab:
            r = _edit_ratio(q, v)
            allow_abbrev = len(q_letters) <= 4 and _is_subseq(q_letters, _letters(v))
            if r <= self.max_edit_ratio or allow_abbrev:
                # Give abbreviation matches a slight distance advantage
                cand.append((r if not allow_abbrev else 0.34, v))

        if not cand:
            # fallback: nearest even if above threshold
            vbest = min(self.vocab, key=lambda v: _edit_ratio(q, v))
            return vbest

        # sort by: distance asc, frequency desc, LONGER tokens first, then lexicographic
        cand.sort(key=lambda rv: (rv[0], -self.freq.get(rv[1], 0), -len(rv[1]), rv[1]))
        neighbors = [v for _, v in cand[:self.k]]

        # weighted majority by global frequency; tie-break to longer token
        best = max(neighbors, key=lambda v: (self.freq.get(v, 0), len(v), v))
        return best

# ====================== numeric parsing / binning ======================
def parse_numeric_or_range(value: Any) -> Optional[Tuple[float,float]]:
    """
    Accepts: 41k-45k, 46K..50K, 41000~45000, 31..35yrs, 48000, "45K"
    Returns (lo, hi) where lo==hi for single values.
    """
    if value is None: return None
    s = _norm_str(value).lower()
    if not s: return None
    s = s.replace("~","..").replace("to","..").replace("—","..").replace("–","..")
    nums = _to_ints(s)
    # handle trailing/leading k (45k -> 45000)
    if re.search(r"\bk\b", s):
        nums = [n*1000 for n in nums]
    if not nums: return None
    if len(nums) == 1:
        n = float(nums[0])
        return (n, n)
    lo, hi = float(min(nums[0], nums[1])), float(max(nums[0], nums[1]))
    return (lo, hi)

def compute_bins(values: List[Tuple[float,float]], strategy:str="quantile", bins:int=5) -> List[float]:
    """
    Produce bin edges (length bins+1).
    - quantile: equal-frequency by midpoints
    - width: equal-width over min..max
    """
    if not values: return [0.0, 1.0]
    mids = sorted((lo+hi)/2.0 for lo,hi in values)
    if strategy == "width":
        lo, hi = float(min(mids)), float(max(mids))
        if hi <= lo: hi = lo + 1.0
        step = (hi - lo) / bins
        return [lo + i*step for i in range(bins)] + [hi]
    # quantile
    edges = [mids[int(round(len(mids)*q)) - 1 if int(round(len(mids)*q)) > 0 else 0]
             for q in [i/bins for i in range(1, bins)]]
    return [mids[0]] + edges + [mids[-1]]

def label_bin(n: float, edges: List[float], suffix:str="") -> str:
    for i in range(len(edges)-1):
        if n <= edges[i+1] or i == len(edges)-2:
            lo, hi = edges[i], edges[i+1]
            if abs(hi) >= 1000 or abs(lo) >= 1000:
                return f"{int(round(lo/1000))}K..{int(round(hi/1000))}K" + suffix
            return f"{int(math.floor(lo))}..{int(math.floor(hi))}" + suffix
    lo, hi = edges[-2], edges[-1]
    return f"{int(lo)}..{int(hi)}" + suffix

# ====================== schema inference ======================
def is_numericish(v: Any) -> bool:
    if isinstance(v, (int,float)): return True
    s = _norm_str(v).lower()
    if not s: return False
    # if any letters exist, treat as non-numeric (we rely on parse success later)
    if any(c.isalpha() for c in s):
        return False
    return bool(_to_ints(s))

def infer_field_roles(rows: List[Dict[str,Any]],
                      include: List[str],
                      exclude: List[str],
                      max_cat_ratio: float=0.3,
                      min_cat_card: int=2,
                      max_group_fields: int=4) -> Tuple[List[str], List[str]]:
    if not rows: return ([],[])
    n = len(rows)
    keys = set()
    for r in rows: keys.update(r.keys())
    keys = [k for k in keys if k not in exclude]

    cat_forced = [k for k in include if k in keys]

    uniques = {k: set() for k in keys}
    has_digit = {k: False for k in keys}
    has_alpha = {k: False for k in keys}
    # track numeric-parse success ratio per field
    parse_ok = {k: 0 for k in keys}
    parse_tot = {k: 0 for k in keys}

    for r in rows:
        for k in keys:
            v = r.get(k)
            if v is None:
                continue
            s = _norm_str(v)
            uniques[k].add(_norm_token(s))
            if any(c.isalpha() for c in s):
                has_alpha[k] = True
            if is_numericish(s):
                has_digit[k] = True
            # try parsing as numeric/range regardless of letters
            pr = parse_numeric_or_range(s)
            parse_tot[k] += 1
            if pr is not None:
                parse_ok[k] += 1

    # numeric if parser succeeds often (>=60%), OR numeric-ish without letters
    numeric_like = set()
    for k in keys:
        ok = parse_ok[k]; tot = max(parse_tot[k], 1)
        if (ok / tot) >= 0.60 or (has_digit[k] and not has_alpha[k]):
            numeric_like.add(k)

    # pick categorical: low-cardinality labels that are NOT numeric_like
    categorical = list(cat_forced)
    for k in keys:
        if k in categorical:
            continue
        card = len(uniques[k])
        if k not in numeric_like and ((min_cat_card <= card <= max(2, int(max_cat_ratio*n))) or has_alpha[k]):
            categorical.append(k)

    # numeric = numeric_like but not already chosen as categorical
    numeric = [k for k in keys if k in numeric_like and k not in categorical]

    # cap number of categorical group fields
    if len(categorical) > max_group_fields:
        categorical = categorical[:max_group_fields]

    return (categorical, numeric)

# ====================== main distill ======================
def distill(rows: Iterable[Dict[str,Any]],
            include_fields: List[str] = None,
            exclude_fields: List[str] = None,
            max_cat_ratio: float = 0.3,
            min_cat_card: int = 2,
            max_group_fields: int = 4,
            k_cat: int = 3,
            max_edit_ratio: float = 0.45,
            null_policy: str = "majority",
            bin_strategy: str = "quantile",
            bins: int = 5,
            count_keys: List[str] = None
            ) -> List[Dict[str,Any]]:
    rows = list(rows or [])
    include_fields = include_fields or []

    # Auto-include common categorical fields if present
    common_labels = ["department", "status", "category", "class", "label"]
    for f in common_labels:
        if f not in include_fields and any(f in r for r in rows):
            include_fields.append(f)

    # default exclude IDs
    default_exclude = {"id","uuid","_id"}
    if exclude_fields:
        exclude_fields = list(set(exclude_fields) | default_exclude)
    else:
        exclude_fields = list(default_exclude)

    count_keys = count_keys or ["count","cnt","weight","freq","frequency"]

    cat_fields, num_fields = infer_field_roles(
        rows, include_fields, exclude_fields,
        max_cat_ratio=max_cat_ratio, min_cat_card=min_cat_card,
        max_group_fields=max_group_fields
    )

    # build KNN majority mappers per categorical field
    cat_knn: Dict[str,KNNMajority] = {}
    for cf in cat_fields:
        tokens = [(r.get(cf, ""), get_count_value(r, count_keys)) for r in rows]
        km = KNNMajority(k=k_cat, max_edit_ratio=max_edit_ratio, null_policy=null_policy)
        km.fit(tokens)
        cat_knn[cf] = km

    # learn bins for numeric fields
    bin_edges: Dict[str, List[float]] = {}
    for nf in num_fields:
        vals = []
        for r in rows:
            pr = parse_numeric_or_range(r.get(nf))
            if pr: vals.append(pr)
        if vals:
            edges = compute_bins(vals, strategy=bin_strategy, bins=bins)
            bin_edges[nf] = edges

    # aggregate
    agg = defaultdict(int)
    for r in rows:
        key_parts = []
        # categorical
        for cf in cat_fields:
            key_parts.append((cf, cat_knn[cf].predict(r.get(cf))))
        # numeric -> binned label
        for nf in num_fields:
            pr = parse_numeric_or_range(r.get(nf))
            if pr and nf in bin_edges:
                mid = (pr[0]+pr[1])/2.0
                key_parts.append((nf, label_bin(mid, bin_edges[nf])))
        if not key_parts:
            continue
        cnt = get_count_value(r, count_keys)
        agg[tuple(key_parts)] += cnt

    # emit rows as dicts; keep order: categorical first, numeric next
    out = []
    for kp, cnt in agg.items():
        d = {}
        for k,v in kp:
            d[k] = v
        d["count"] = cnt
        out.append(d)

    out.sort(key=lambda x: [x.get(k,"") for k in cat_fields + num_fields] + [x["count"]])
    return out

# ====================== entrypoint ======================
def _parse_args(argstr: str) -> Dict[str,Any]:
    cfg = {}
    for tok in (argstr or "").split():
        if "=" not in tok: continue
        k,v = tok.split("=",1)
        if k=="include": cfg["include_fields"] = [s.strip() for s in v.split(",") if s.strip()]
        elif k=="exclude": cfg["exclude_fields"] = [s.strip() for s in v.split(",") if s.strip()]
        elif k=="k": cfg["k_cat"] = int(v)
        elif k=="bins": cfg["bins"] = int(v)
        elif k=="bin": cfg["bin_strategy"] = v
        elif k=="max_edit": cfg["max_edit_ratio"] = float(v)
        elif k=="null": cfg["null_policy"] = v
        elif k=="max_cat_ratio": cfg["max_cat_ratio"] = float(v)
        elif k=="min_cat_card": cfg["min_cat_card"] = int(v)
        elif k=="max_group_fields": cfg["max_group_fields"] = int(v)
        elif k=="counts": cfg["count_keys"] = [s.strip() for s in v.split(",") if s.strip()]
    return cfg

def main(input_path: str, args: str = "") -> str:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = _parse_args(args)
    result = distill(data if isinstance(data, list) else [], **cfg)
    return json.dumps(result, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import sys
    print(main(sys.argv[1] if len(sys.argv)>1 else "-", " ".join(sys.argv[2:])))
