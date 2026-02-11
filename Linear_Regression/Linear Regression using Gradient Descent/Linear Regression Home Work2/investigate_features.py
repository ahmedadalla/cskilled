
def investigate(df):
    import matplotlib.pyplot as plt
    for col in df.columns:
        plt.scatter(df[col],df.iloc[:,-1])
        plt.title(col)
        plt.show()