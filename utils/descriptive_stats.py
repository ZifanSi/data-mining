import json, sys, math

def midpoints(interval):
    a,b=map(int,interval.split("-"))
    return (a+b)/2

def stats(data):
    n=sum(d["freq"] for d in data)
    mids=[midpoints(d["class"]) for d in data]
    freqs=[d["freq"] for d in data]

    # mean
    mean=sum(f*m for f,m in zip(freqs,mids))/n

    # variance, std dev
    var=sum(f*(m**2) for f,m in zip(freqs,mids))/n - mean**2
    std=math.sqrt(var)

    # cumulative freq
    cum=[sum(freqs[:i+1]) for i in range(len(freqs))]

    # median class
    half=n/2
    for i,c in enumerate(cum):
        if c>=half:
            L=int(data[i]["class"].split("-")[0])
            h=int(data[i]["class"].split("-")[1]) - L
            CF=c-freqs[i]
            f=freqs[i]
            median=L+((half-CF)/f)*h
            break

    # mode (empirical relation)
    mode=3*median-2*mean

    # skewness
    skew=(mean-median)/std if std else 0

    return {"n":n,"mean":mean,"median":median,"mode":mode,
            "variance":var,"std_dev":std,"skewness":skew}

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python descriptive_stats.py db/age_freq.json"); sys.exit(1)
    data=json.load(open(sys.argv[1]))
    result=stats(data)
    for k,v in result.items():
        print(f"{k}: {v:.2f}")
