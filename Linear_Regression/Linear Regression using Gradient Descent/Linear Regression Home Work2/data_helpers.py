from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd
def load_data(path,preprocessing):
    df=pd.read_csv(path)
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if preprocessing == 1:
        scaller=MinMaxScaler()
        x = scaller.fit_transform(x)
    elif preprocessing ==2:
        scaller=StandardScaler()
        x = scaller.fit_transform(x)
    elif preprocessing==0:
        x = df.iloc[:, :-1]
    return df,x,y

def add_bias_columns(X):
    import numpy as np
    X=np.hstack([np.ones((X.shape[0],1)),X])
    return X