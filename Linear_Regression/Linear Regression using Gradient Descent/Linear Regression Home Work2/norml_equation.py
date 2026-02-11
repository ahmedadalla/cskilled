import numpy as np

def normal_equation_linear_regression(X:np.array,Y):
    x_t=X.T
    W = np.linalg.inv((x_t.dot(X))).dot(x_t.dot(Y))
    return W