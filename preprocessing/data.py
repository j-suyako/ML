import numpy as np
from utils.validation import check_array

def scale(X, axis=0):
    X = check_array(X)
    mean_ = np.mean(X, axis)
    scale_ = np.std(X, axis)
    return (X - mean_) / scale_
