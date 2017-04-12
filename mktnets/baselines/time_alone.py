from statsmodels.tsa import arima_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, math
import statsmodels.tsa.stattools as st

if __name__ == "__main__":

    # Read data.
    df_path = sys.path[0] + "/uber.csv"
    df = pd.read_csv(df_path)

    # Data standardization.
    df = df.groupby("group_ts").sum()
    df["timestamp"] = df.index
    threshold = df["timestamp"].quantile(0.85)
    df.index = pd.to_datetime(df["timestamp"] - 8 * 3600, unit = "s")
    """
    uber_scaler = StandardScaler().fit(df[["uber_sup"]])
    df["uber_sup"] = uber_scaler.transform(df[["uber_sup"]])
    """

    # ARIMA (Time -> Uber).
    train = df[df["timestamp"] <= threshold]["uber_sup"]
    test = df[df["timestamp"] >= threshold - 86400]["uber_sup"]

    # Resampling to get hourly data (otherwise exception).
    uber_train = train.resample('H').ffill(). \
    reindex(pd.date_range(train.index[0], train.index[-1], freq = "H"))
    uber_test = test.resample('H').ffill(). \
    reindex(pd.date_range(test.index[0], test.index[-1], freq = "H"))

    # Fit model and predict.
    arima = arima_model.ARIMA(uber_train, (15, 0, 5)).fit()
    uber_predict = arima.predict(uber_test.index[0], uber_test.index[-1], dynamic = True)
    mse = math.sqrt(np.mean((uber_predict - uber_test) / 24) ** 2)
    print "Root Mean squared error: " + str(mse)

    """
    # Plot figures.
    fig_path = sys.path[0] + "/fig/time_alone_reg.png"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.set_xlabel("Time", fontsize = 10)
    ax.set_ylabel("Value", fontsize = 10)
    ax.plot(df["uber_data"], c = "r", label = "real")
    ax.plot(arima.fittedvalues[:-24], c = "k", label = "train")
    ax.plot(uber_predict, c = "b", label = "test")
    ax.set_ylim([-2, 4])
    ax.set_xlim(["2016-11-12", "2016-12-22"])
    leg = plt.legend(loc = 2, ncol = 3, fontsize = 10, borderaxespad=0.)
    leg.get_frame().set_alpha(0)
    plt.savefig(fig_path, bbox_inches = "tight")

    fig_path = sys.path[0] + "/fig/time_alone_acc.png"
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlabel("True Value", fontsize = 10)
    ax.set_ylabel("Prediction", fontsize = 10)
    ax.scatter(uber_test, uber_predict, c = "b", s = 2, alpha = 0.5)
    ax.plot([-2, 2.5], [-2, 2.5], c = "k", linewidth = 2)
    plt.savefig(fig_path, bbox_inches = "tight")
    """
