import pandas as pd
import numpy as np
import sys, time, h5py

# Return a matrix representation.
def get_matrix(vector, each_row = 29):
    matrix = []
    while len(vector) > 0:
        matrix.append(vector[:each_row])
        vector = vector[each_row:]
    return matrix

# Return a tensor representation.
def get_tensor(df):
    sup = get_matrix(df.sort_values("geo_id")["uber_sup"].values)
    dem = get_matrix(df.sort_values("geo_id")["uber_dem"].values)
    return [sup, dem]
    
# Main.
if __name__ == "__main__":

    # Read data.
    f = pd.read_csv(sys.path[0] + "/uber.csv")
    f = f[f["group_ts"]< 1482393600]

    # Get tensor format data.
    h5_data = []
    h5_date = []
    for ts, df in f.groupby("group_ts"):
        t = time.localtime(ts - 10800)
        h5_date.append(time.strftime("%Y%m%d", t) + "{:02d}".format(t.tm_hour + 1))
        h5_data.append(get_tensor(df))

    # Save H5 data.
    h5f = h5py.File(sys.path[0] + "/uber.h5", "w")
    h5f.create_dataset("data", data = h5_data)
    h5f.create_dataset("date", data = h5_date)
    h5f.close()
