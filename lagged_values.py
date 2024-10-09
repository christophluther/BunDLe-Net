# create datasets with lagged values of neurons and behaviour

import numpy as np
import pandas as pd

# load data

# for final layer, simply use df[768:]
filename = "test_oct_2"

X = pd.read_csv(f"data_2/racetrack-loop-highway/sac/{filename}_neurons.csv").to_numpy()
B = pd.read_csv(f"data_2/racetrack-loop-highway/sac/{filename}_actions.csv").to_numpy()

# shift indices for lagged values
def shift_k(x, b, k=1):
    """Shift indices of labels"""
    return x[:-k], b[k:]

# lag by time
def lag_k(x, b, k=1, mean=False):
    """You want neurons in past k steps to predict label in t
        -concatenate neurons from t-k to t-1
        -average neurons from t-k to t-1
    """
    # create new dataset with concatenated neurons
    if not mean:
        x_ = np.concatenate([x[i:-k+i] for i in range(k)], axis=1)
    else:
        x_ = np.zeros((x.shape[0]-k, x.shape[1]))
        for i in range(k):
            x_ += x[i:-k+i]
        x_ /= k
    return x_, b[k:]
