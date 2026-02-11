import argparse
import matplotlib.pyplot as plt
import numpy as np
from data_helpers import load_data,add_bias_columns
from gradient_decent_linear_reg import gradient_descent_linear_regression
from regresion_helpers import train,predict,visualize_cost
from optimization import optimize
from investigate_features import  investigate
from norml_equation import normal_equation_linear_regression


parser=argparse.ArgumentParser(description="linear regression")
parser.add_argument('--data',type=str,default='data/dataset_200x4_regression.csv',help='data location')
parser.add_argument('--preprocessing' ,default=1,type=int,help='0 for non'
                                                                            '1 for minmax'
                                                                            '2 for standard'
                                                                            )
parser.add_argument('--choice',type=int,default=3,help='0 for verification'
                                                                    '1 train with all features'
                                                                    '2 train with best feature'
                                                                    '3 normal equation')

parser.add_argument('--step_size',type=float,help='step size',default=0.01)
parser.add_argument('--precision',type=float,help='precision',default=0.00001)
parser.add_argument('--max_iter',type=int,help='step size',default=10000)


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
    #w=np.array([1.,1.,1.,1.])
    optimal_w,costH=train(x,y,max_iter=args.max_iter,step_size=args.step_size,precision=args.precision)
    y_pred=predict(x,optimal_w)

elif args.choice==2:
    x=x[:,0]
    x=x.reshape((-1,1))
    optimal_w, costH = train(x, y, max_iter=args.max_iter, step_size=args.step_size, precision=args.precision)
    y_pred = predict(x, optimal_w)
    #visualize_cost(costH)

elif args.choice==3:
    optimal_w_normal=normal_equation_linear_regression(x,y)
    optimal_w_gradient, costH = train(x, y, max_iter=args.max_iter, step_size=args.step_size, precision=args.precision)
    y_pred_normal = predict(x, optimal_w_normal)
    y_pred_gradient=predict(x,optimal_w_gradient)
    import matplotlib.pyplot as plt
    #plt.scatter(y,y_pred_normal,color='green')
    #plt.scatter(y,y_pred_gradient,color='red')
    #plt.xlabel('ground truth')
    #plt.ylabel("normal equation")
    #plt.title("predictions of gradient decent vs normal equation")
    #plt.show()

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))

    # Scatter plots with labels
    plt.scatter(y, y_pred_normal, color='green', alpha=0.7, label='Normal Equation')
    plt.scatter(y, y_pred_gradient, color='red', alpha=0.7, label='Gradient Descent')

    # Perfect prediction line (very important for comparison)
    plt.plot([min(y), max(y)], [min(y), max(y)],
             color='blue', linestyle='--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Ground Truth', fontsize=12)
    plt.ylabel('Predictions', fontsize=12)
    plt.title('Models Predictions vs Ground Truth', fontsize=14, fontweight='bold')

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()





