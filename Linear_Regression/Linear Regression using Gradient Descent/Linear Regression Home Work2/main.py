import argparse
import matplotlib.pyplot as plt
import numpy as np
from data_helpers import load_data,add_bias_columns
from gradient_decent_linear_reg import gradient_descent_linear_regression

parser=argparse.ArgumentParser(description="linear regression")


parser.add_argument('--data',type=str,default='data/dataset_200x4_regression.csv',help='data location')
parser.add_argument('--preprocessing' ,default=1,type=int,help='0 for non'
                                                                            '1 for minmax'
                                                                            '2 for standard')
parser.add_argument('--choice',type=int,default=1,help='0 for verification'
                                                                    '1 train with all features')

parser.add_argument('--step_size',type=float,help='step size',default=0.01)
parser.add_argument('--precision',type=float,help='precision',default=0.0001)
parser.add_argument('--max_iter',type=int,help='step size',default=1000)


args=parser.parse_args()

if args.choice==0:
    x=np.array([0,0.2,0.4,0.8,1])
    y= x + 5
    x_vis=x.copy()
    x=x.reshape((-1,1))
    x=np.hstack([np.ones((x.shape[0],1)),x])

    new_w,cost=gradient_descent_linear_regression(x,y)

    pred_y=np.dot(x,new_w)

    plt.scatter(x_vis,y, color='red')
    plt.plot(x_vis,pred_y)
    plt.show()
else:
    df,x,y=load_data(args.data,args.preprocessing)
    x = add_bias_columns(x)
if args.choice==1:
    w=np.array([1.,1.,1.,1.])
    optimal_w,costH=gradient_descent_linear_regression(x,y,w,max_iter=args.max_iter,step_size=args.step_size,precision=args.precision)

