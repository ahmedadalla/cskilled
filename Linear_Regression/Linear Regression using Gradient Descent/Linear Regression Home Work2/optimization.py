from gradient_decent_linear_reg import gradient_descent_linear_regression


def optimize(X,Y):
    stepSize={0.1,0.001,0.0001,0.00001,0.0000001}
    precision={0.1,0.001,0.0001,0.00001}
    minCost=float('inf')
    itnum=0
    combination={"stepSize":0,
                 "precision":0}
    weights=0
    for step in stepSize:
        for p in precision:
            for i in range (3):
                W,costs=gradient_descent_linear_regression(X,Y,precision=p,step_size=step,max_iter=10000)
                if costs[-1]<minCost:
                    weights=W
                    minCost=costs[-1]
                    itnum=len(costs)
                    combination['stepSize']=step
                    combination['precision']=p
    print(f"the minimum cost = {minCost}\nnumber of iteration = {itnum}\nbest combination is step size= {combination['stepSize']} ,precision ={combination['precision']}")
    return weights,minCost,itnum,combination

