import sys,json

def train(train,epochs=200,lr=0.01,C=1.0):
    w=[0.0]*len(train[0]["x"]); b=0.0
    for _ in range(epochs):
        for r in train:
            x,y=(r["x"],1 if r["y"]==1 else -1)
            cond=y*(sum(wi*xi for wi,xi in zip(w,x))+b)
            if cond>=1: # no loss
                w=[wi-lr*wi for wi in w]
            else: # hinge loss
                w=[wi-lr*(wi-C*y*xi) for wi,xi in zip(w,x)]
                b+=lr*C*y
    return w,b

def predict(x,w,b): return 1 if sum(wi*xi for wi,xi in zip(w,x))+b>=0 else -1

if __name__=="__main__":
    data=json.load(open(sys.argv[1]))
    q=json.loads(sys.argv[2])
    w,b=train(data)
    print("Predict:",predict(q,w,b))
