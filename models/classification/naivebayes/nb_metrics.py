def accuracy(y_true,y_pred):
    n=len(y_true); return sum(yt==yp for yt,yp in zip(y_true,y_pred))/n if n else 0.0
