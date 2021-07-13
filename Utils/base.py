import numpy as np


def normalize(arr: np.ndarray, multi: bool = False) -> np.ndarray:
    if multi:
        return np.array([(arr[:, i] - np.min(arr[:, i])) / (np.max(arr[:, i]) - np.min(arr[:, i])) for i in
                         range(arr.shape[1])]).T
    else:
        return np.around((arr - np.min(arr)) / (np.max(arr) - np.min(arr)).T, 3)


def running_mean(arr: np.ndarray, num: int) -> np.ndarray:
    cum_sum = np.cumsum(np.insert(arr=arr, values=[np.mean(arr)] * num, obj=0))
    return (cum_sum[num:] - cum_sum[:-num]) / float(num)


def cum_mean(arr: np.ndarray) -> np.ndarray:
    cum_sum = np.cumsum(arr, axis=0)
    for i in range(cum_sum.shape[0]):
        cum_sum[i] = cum_sum[i] / (i + 1)
    return cum_sum