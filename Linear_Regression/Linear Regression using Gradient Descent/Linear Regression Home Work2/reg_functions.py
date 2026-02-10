import numpy as np
from scipy.optimize import check_grad

def MSE(W,X,Y):
    y_pred=np.dot(X,W)
    error=(y_pred-Y)**2
    return  np.sum(error)/(X.shape[0]*2)

def MSE_Drev(W,X,Y):
    y_pred = np.dot(X, W)
    error = (y_pred - Y)
    return X.T @ error / X.shape[0]

if __name__=="__main__":
    X = np.array([0, 0.2, 0.4, 0.8, 1.0])
    Y = X + 5
    W = np.array([1.0, 1.0])
    X = X.reshape((-1, 1))
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    error = check_grad(MSE, MSE_Drev, W, X, Y)
    print("Gradient check error:", error)


