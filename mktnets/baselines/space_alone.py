from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, math

if __name__ == "__main__":

    # Read data.
    df_path = sys.path[0] + "/uber.csv"
    df = pd.read_csv(df_path)

    # Data standardization.
    """
    for col in ["uber_sup", "uber_dem"]:
        df[col] = ((df[col] - df[col].min()) / (df[col].max() - df[col].min())) * 2 - 1
    """

    # KNN (Neighbor -> Uber).
    knn = neighbors.KNeighborsRegressor(4, "distance")
    test_list = []
    predict_list = []
    for time, data in df.groupby("group_ts"):
        train = data[60:]
        test = data[:60]
        uber_train = train["uber_sup"].values
        loc_train = train[["id_lat", "id_lng"]].values
        uber_test = test["uber_sup"].values
        loc_test = test[["id_lat", "id_lng"]].values
        uber_predict = knn.fit(loc_train, uber_train).predict(loc_test)
        test_list.extend(uber_test)
        predict_list.extend(uber_predict)
    for time, data in df.groupby("group_ts"):
        train = data[60:]
        test = data[:60]
        uber_train = train["uber_dem"].values
        loc_train = train[["id_lat", "id_lng"]].values
        uber_test = test["uber_dem"].values
        loc_test = test[["id_lat", "id_lng"]].values
        uber_predict = knn.fit(loc_train, uber_train).predict(loc_test)
        test_list.extend(uber_test)
        predict_list.extend(uber_predict)
    rmse = math.sqrt(np.mean((np.array(predict_list) - np.array(test_list)) ** 2))
    print "Root Mean squared error: " + str(rmse)

    """
    # Plot figures.
    fig_path = sys.path[0] + "/fig/space_alone_acc.png"
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlabel("True Value", fontsize = 10)
    ax.set_ylabel("Prediction", fontsize = 10)
    ax.scatter(test_list, predict_list,
        c = "b", s = 2, alpha = 0.5)
    ax.plot([-1, 20], [-1, 20], c = "k", linewidth = 2)
    plt.savefig(fig_path, bbox_inches = "tight")
    """
