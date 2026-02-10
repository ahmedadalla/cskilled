import numpy as np

from reg_functions import *

def gradient_descent_linear_regression(X, t,w=None, step_size = 0.1, precision = 0.0001, max_iter = 1000):
    if w is not None:
        cur_w=w
    else:
        cur_w=np.random.rand(X.shape[1])
    last_w=cur_w+100*precision
    it = 0
    costH=[]
    while np.linalg.norm(cur_w - last_w) > precision and it < max_iter:
        last_w = cur_w.copy()
        gradient = MSE_Drev(cur_w,X,t)
        cost=MSE(cur_w,X,t)
        costH.append(cost)

        cur_w -= gradient * step_size
        #print(f'w >> {last_w}  cost >> {cost} gradient >> {gradient}')
        it += 1

    return cur_w,costH

