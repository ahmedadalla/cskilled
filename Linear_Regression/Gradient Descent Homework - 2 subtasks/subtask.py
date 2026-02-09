import numpy as np
from  gradient_decent_vec import gradient_descent


def f(array):
    return np.sin(array[0]) + np.cos(array[1]) + np.sin(array[2])

def f_drev(array):
    x=array[0]
    y=array[1]
    z=array[2]

    f_d_x= np.cos(x)
    f_d_y= -np.sin(y)
    f_d_z= np.cos(z)

    return np.array([f_d_x,f_d_y,f_d_z])

if __name__=='__main__':
    intial_start=np.array([1.,2.,3.5])
    ends=gradient_descent(f_drev,intial_start)
    print(f'ends at > {ends}')
    print(f'with value  = {f(ends)}')