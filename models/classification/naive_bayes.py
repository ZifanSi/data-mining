# models/classification/naive_bayes.py
import sys, json, math, argparse
from collections import defaultdict, Counter
from typing import Any, Dict, Iterable, List, Tuple, Optional, Union

Number = Union[int, float]

# ---------- utils: flatten + numeric helpers ----------
def _is_number_like(x: Any) -> bool:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return True
    if isinstance(x, str):
        try:
            float(x.strip())
            return True
        except ValueError:
            return False
    return False

def _to_number(x: Any) -> Optional[Number]:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except ValueError:
            return None
    return None

def _flatten(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _flatten(f"{prefix}[{i}]" if prefix else f"[{i}]", v, out)
    else:
        if prefix:  # ignore empty top-level if someone passes a scalar
            out[prefix] = value

def flatten_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    _flatten("", obj, out)
    return out

# ---------- model ----------
class NaiveBayesJSON:
    """
    General Naive Bayes for arbitrary JSON records.
    - Auto-detects feature types (numeric/categorical)
    - Categorical: Laplace smoothing
    - Numeric: Gaussian likelihood with variance floor
    - Supports per-sample weights
    """
    def __init__(self, laplace: float = 1.0, var_floor: float = 1e-9):
        self.laplace = laplace
        self.var_floor = var_floor
        self.classes_: List[Any] = []
        self.class_weight_: Counter = Counter()
        self.n_weight_: float = 0.0

        self.feature_types_: Dict[str, str] = {}  # 'categorical' | 'numeric'
        self.cat_counts_: Dict[str, Dict[Any, Counter]] = defaultdict(lambda: defaultdict(Counter))
        self.cat_totals_: Dict[str, Dict[Any, float]] = defaultdict(lambda: defaultdict(float))
        self.cat_cardinality_: Dict[str, int] = {}

        # numeric params: feature -> class -> (mean, var, weight_sum)
        self.num_params_: Dict[str, Dict[Any, Tuple[float, float, float]]] = defaultdict(dict)
        self.log_priors_: Dict[Any, float] = {}
        self._feat_numeric_votes_: Counter = Counter()
        self._feat_total_votes_: Counter = Counter()

    def _decide_feature_types(self):
        for feat, total in self._feat_total_votes_.items():
            num_votes = self._feat_numeric_votes_[feat]
            self.feature_types_[feat] = 'numeric' if num_votes >= (total - num_votes) else 'categorical'

    def fit(
        self,
        X: Iterable[Dict[str, Any]],
        y: Iterable[Any],
        sample_weight: Optional[Iterable[float]] = None
    ):
        X = list(X)
        y = list(y)
        if sample_weight is None:
            sample_weight = [1.0] * len(X)
        else:
            sample_weight = [float(w) for w in sample_weight]

        flat_X = [flatten_json(rec) for rec in X]

        # vote feature types
        for feat_row in flat_X:
            for k, v in feat_row.items():
                self._feat_total_votes_[k] += 1
                if _is_number_like(v):
                    self._feat_numeric_votes_[k] += 1
        self._decide_feature_types()

        # accumulate per-class stats
        num_stats: Dict[str, Dict[Any, List[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))
        cat_vocab: Dict[str, set] = defaultdict(set)

        for feat_row, cls, w in zip(flat_X, y, sample_weight):
            self.class_weight_[cls] += w
            self.n_weight_ += w

            for k, v in feat_row.items():
                if self.feature_types_[k] == 'numeric':
                    val = _to_number(v)
                    if val is None:
                        continue
                    s = num_stats[k][cls]  # [sum, sumsq, wsum]
                    s[0] += w * val
                    s[1] += w * (val * val)
                    s[2] += w
                else:
                    sval = str(v)
                    self.cat_counts_[k][cls][sval] += w
                    self.cat_totals_[k][cls] += w
                    cat_vocab[k].add(sval)

        # finalize
        self.classes_ = list(self.class_weight_.keys())
        self.log_priors_ = {c: math.log(self.class_weight_[c] / self.n_weight_) for c in self.classes_}

        for feat, by_class in num_stats.items():
            for cls, (sum_, sumsq, wsum) in by_class.items():
                if wsum <= 0:
                    continue
                mean = sum_ / wsum
                var = max((sumsq / wsum) - (mean * mean), self.var_floor)
                self.num_params_[feat][cls] = (mean, var, wsum)

        for feat, vocab in cat_vocab.items():
            self.cat_cardinality_[feat] = max(1, len(vocab))

        return self

    def _log_gauss(self, x: float, mean: float, var: float) -> float:
        return -0.5 * (math.log(2 * math.pi * var) + ((x - mean) ** 2) / var)

    def predict_proba_one(self, rec: Dict[str, Any]) -> Dict[Any, float]:
        flat = flatten_json(rec)
        logp = {c: self.log_priors_[c] for c in self.classes_}

        for feat, ftype in self.feature_types_.items():
            if feat not in flat:
                continue

            if ftype == 'numeric':
                x = _to_number(flat[feat])
                if x is None:  # incompatible type -> ignore
                    continue
                for c in self.classes_:
                    params = self.num_params_.get(feat, {}).get(c)
                    if not params:
                        continue
                    mean, var, _ = params
                    logp[c] += self._log_gauss(x, mean, var)
            else:
                sval = str(flat[feat])
                V = self.cat_cardinality_.get(feat, 1)
                for c in self.classes_:
                    count = self.cat_counts_[feat][c][sval]
                    total = self.cat_totals_[feat][c]
                    # Laplace smoothing
                    if total > 0:
                        prob = (count + 1.0) / (total + V * 1.0)
                    else:
                        prob = 1.0 / V
                    logp[c] += math.log(prob)

        # normalize
        m = max(logp.values())
        exps = {c: math.exp(v - m) for c, v in logp.items()}
        Z = sum(exps.values()) or 1.0
        return {c: exps[c] / Z for c in self.classes_}

    def predict_proba(self, X: Iterable[Dict[str, Any]]) -> List[Dict[Any, float]]:
        return [self.predict_proba_one(r) for r in X]

# ---------- io helpers ----------
def load_records(path: str) -> List[Dict[str, Any]]:
    # try JSON array; fallback to JSONL
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError("Top-level JSON must be a list of objects.")
    except json.JSONDecodeError:
        recs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        return recs

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="General Naive Bayes for arbitrary JSON records. Outputs pure probabilities."
    )
    ap.add_argument("train", help="Training JSON/JSONL (list of objects)")
    ap.add_argument("predict", nargs="?", help="Prediction JSON/JSONL (optional; defaults to training set)")
    ap.add_argument("--target", default="target", help="Name of label field in records (default: target)")
    ap.add_argument("--weight", default=None, help="Optional weight field (e.g., count)")
    ap.add_argument("--laplace", type=float, default=1.0, help="Laplace smoothing for categorical (default 1.0)")
    ap.add_argument("--var_floor", type=float, default=1e-9, help="Variance floor for numeric (default 1e-9)")
    args = ap.parse_args()

    train = load_records(args.train)
    if not train:
        print("[]")
        return

    # labels
    if not all(isinstance(r, dict) and (args.target in r) for r in train):
        sys.stderr.write(f"ERROR: training records must include '{args.target}'.\n")
        sys.exit(1)

    y = [r[args.target] for r in train]
    X = [{k: v for k, v in r.items() if k != args.target} for r in train]

    # sample weights
    if args.weight is not None:
        w = [float(r.get(args.weight, 1.0)) for r in train]
    else:
        w = None

    nb = NaiveBayesJSON(laplace=args.laplace, var_floor=args.var_floor).fit(X, y, sample_weight=w)

    # prediction set
    pred_set = load_records(args.predict) if args.predict else X
    probs_list = nb.predict_proba(pred_set)

    # pure probabilities, one JSON per line
    for probs in probs_list:
        print(json.dumps(probs, ensure_ascii=False))

if __name__ == "__main__":
    main()
