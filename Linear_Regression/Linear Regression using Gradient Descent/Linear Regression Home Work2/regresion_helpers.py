from gradient_decent_linear_reg import gradient_descent_linear_regression
import numpy as np
def train(x,y,w=None,step_size=0.001,precision=0.00001,max_iter=10000):
    optimal_w,cost_history=gradient_descent_linear_regression(x, y, w=w, max_iter=max_iter, step_size=step_size,precision=precision)
    return optimal_w,cost_history
def predict(x,w):
    y_pred=np.dot(x,w)
    return y_pred

def visualize_cost(cost_history):
    import matplotlib.pyplot as plt
    plt.plot(list(range(len(cost_history))),cost_history,color='red',ls='--')
    plt.xlabel('iteration')
    plt.ylabel("cost")
    plt.show()