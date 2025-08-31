import sys,json,math

def sigmoid(z): return 1/(1+math.exp(-z))

def train(train,epochs=200,lr=0.1):
    w=[0.0]*len(train[0]["x"]); b=0.0
    for _ in range(epochs):
        for r in train:
            x,y=r["x"],r["y"]
            z=sum(wi*xi for wi,xi in zip(w,x))+b
            p=sigmoid(z)
            err=y-p
            w=[wi+lr*err*xi for wi,xi in zip(w,x)]
            b+=lr*err
    return w,b

def predict(x,w,b): return 1 if sigmoid(sum(wi*xi for wi,xi in zip(w,x))+b)>=0.5 else 0

if __name__=="__main__":
    data=json.load(open(sys.argv[1]))
    q=json.loads(sys.argv[2])
    w,b=train(data)
    print("Predict:",predict(q,w,b))
