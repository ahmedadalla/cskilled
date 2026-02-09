import numpy as np



def gradient_descent(fderiv, inital_start,step_size=0.001,precision=0.00001,max_iter=1000):
    cur_start= inital_start
    last_x=cur_start + 1 *precision
    it=0
    while np.linalg.norm(cur_start-last_x) > precision and it < max_iter:
        last_x=cur_start.copy()
        gradient=fderiv(cur_start)
        cur_start-=gradient*step_size
        it+=1
    return  cur_start


