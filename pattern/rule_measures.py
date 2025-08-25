import json, sys

def rule_measures(data):
    n = data["transactions"]
    A = data["counts"]["A"]
    B = data["counts"]["B"]
    AB = data["counts"]["AB"]

    # probabilities
    pA, pB, pAB = A/n, B/n, AB/n
    pNotB = 1 - pB
    pA_notB = (A - AB) / n

    # Support & Confidence
    support = pAB
    confidence = pAB / pA if pA else 0

    # Lift
    lift = (pAB / (pA * pB)) if pA and pB else 0

    # Leverage
    leverage = pAB - pA * pB

    # Conviction
    conviction = (pA * pNotB / pA_notB) if pA_notB > 0 else float("inf")

    # Odds Ratio
    odds_B_given_A = (pAB / pA_notB) if pA_notB > 0 else float("inf")
    odds_B_given_notA = ((pB - pAB) / (1 - pA - pB + pAB)) if (1 - pA - pB + pAB) > 0 else float("inf")
    odds_ratio = (odds_B_given_A / odds_B_given_notA) if odds_B_given_notA > 0 else float("inf")

    return {
        "support": support,
        "confidence": confidence,
        "lift": lift,
        "leverage": leverage,
        "conviction": conviction,
        "odds_ratio": odds_ratio
    }

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python rule_measures.py db/rules.json"); sys.exit(1)
    data=json.load(open(sys.argv[1]))
    result=rule_measures(data)
    for k,v in result.items():
        print(f"{k}: {v:.4f}")
