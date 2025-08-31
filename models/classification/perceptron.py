import sys, json

def perceptron(train, epochs=100, lr=1.0):
    w=[0.0]*len(train[0]["x"]); b=0.0
    for _ in range(epochs):
        for r in train:
            x,y=r["x"],1 if r["y"]==1 else -1
            pred=1 if sum(wi*xi for wi,xi in zip(w,x))+b>=0 else -1
            if pred!=y:
                w=[wi+lr*y*xi for wi,xi in zip(w,x)]
                b+=lr*y
    return w,b

def predict(x,w,b): return 1 if sum(wi*xi for wi,xi in zip(w,x))+b>=0 else -1

if __name__=="__main__":
    data=json.load(open(sys.argv[1]))
    q=json.loads(sys.argv[2])
    w,b=perceptron(data)
    print("Predict:",predict(q,w,b))
