import numpy as np

def check_array(X):
    if hasattr(X, '__array__'):
        return np.asarray(X)
    elif isinstance(X, list):
        return np.array(X)
    else:
        raise ValueError("Array-like needed, 'X' got %s" % type(X))


def Check_X_y(X, y):
    X = check_array(X)
    y = check_array(y)
    return X, y


def _num_samples(x):
    """返回类array数组的数组大小"""
    if hasattr(x, 'fit') and callable(x.fit):
        raise TypeError('Expected sequence or array-like, got estimator %s' % type(x))

    if not hasattr(x, '__len__') or hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Excepted sequence or array-like, got %s" % type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)