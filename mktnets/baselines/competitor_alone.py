from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, math

if __name__ == "__main__":

    # Read data.
    df_path = sys.path[0] + "/uber.csv"
    df = pd.read_csv(df_path)

    # Data standardization.
    #uber_scaler = StandardScaler().fit(df[["uber_num"]])
    #uber_data = uber_scaler.transform(df[["uber_num"]])
    #lyft_scaler = StandardScaler().fit(df[["lyft_num"]])
    #lyft_data = lyft_scaler.transform(df[["lyft_num"]])
    uber_train = df["uber_sup"][:1000].values.reshape(-1, 1)
    uber_test = df["uber_sup"][1000:].values.reshape(-1, 1)
    lyft_train = df["lyft_sup"][:1000].values.reshape(-1, 1)
    lyft_test = df["lyft_sup"][1000:].values.reshape(-1, 1)

    # Linear regression (Lyft -> Uber).
    regr = linear_model.LinearRegression()
    regr.fit(lyft_train, uber_train)
    uber_predict = regr.predict(lyft_test)
    mse = math.sqrt(np.mean((uber_predict - uber_test) ** 2))
    print "Root Mean squared error: " + str(mse)

    """
    # Plot figures.
    fig_path = sys.path[0] + "/competitor_alone_reg.png"
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlabel("Lyft", fontsize = 10)
    ax.set_ylabel("Uber", fontsize = 10)
    ax.scatter(lyft_train, uber_train,
        c = "r", s = 2, alpha = 0.5, label = "train")
    ax.scatter(lyft_test, uber_test,
        c = "b", s = 2, alpha = 0.5, label = "test")
    px = [-1, 20]
    py = [float(regr.predict(-1)), float(regr.predict(20))]
    ax.plot(px, py, c = "k", linewidth = 2)
    leg = plt.legend(loc = 2, fontsize = 10, borderaxespad=0.)
    leg.get_frame().set_alpha(0)
    plt.savefig(fig_path, bbox_inches = "tight")

    fig_path = sys.path[0] + "/competitor_alone_acc.png"
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlabel("True Value", fontsize = 10)
    ax.set_ylabel("Prediction", fontsize = 10)
    ax.scatter(uber_test, uber_predict,
        c = "b", s = 2, alpha = 0.5)
    ax.plot([-1, 20], [-1, 20], c = "k", linewidth = 2)
    plt.savefig(fig_path, bbox_inches = "tight")
    """
