"""Various calculations preformed on arrays.

Usage:
 ./base.py

Author:
 Peter Rigali - 2021-08-30
"""
from typing import Optional
import numpy as np


def normalize(arr: np.ndarray, multi: Optional[bool] = None) -> np.ndarray:
    """

    Normalize an Array.

    :param arr: Input array.
    :type arr: np.ndarray
    :param multi: If array has multiple columns, default is None.
    :type multi: bool
    :return: Normalized array.
    :rtype: np.ndarray
    :example: *None*
    :note: Set *multi* to True, if multiple columns.

    """
    if multi:
        return np.array([(arr[:, i] - np.min(arr[:, i])) / (np.max(arr[:, i]) - np.min(arr[:, i])) for i in
                         range(arr.shape[1])]).T
    else:
        return np.around((arr - np.min(arr)) / (np.max(arr) - np.min(arr)).T, 3)


def running_mean(arr: np.ndarray, num: int) -> np.ndarray:
    """

    Calculate the running mean on *num* interval

    :param arr: Input array.
    :type arr: np.ndarray
    :param num: Input int, default is 50.
    :type num: int
    :return: Running mean for a given array.
    :rtype: np.ndarray
    :example: *None*
    :note: *None*

    """
    cum_sum = np.cumsum(np.insert(arr=arr, values=[np.mean(arr)] * num, obj=0))
    return (cum_sum[num:] - cum_sum[:-num]) / float(num)


def cumulative_mean(arr: np.ndarray) -> np.ndarray:
    """

    Calculate the cumulative mean.

    :param arr: Input array.
    :type arr: np.ndarray
    :return: Cumulative mean for a given array.
    :rtype: np.ndarray
    :example: *None*
    :note: *None*

    """
    cum_sum = np.cumsum(arr, axis=0)
    for i in range(cum_sum.shape[0]):
        cum_sum[i] = cum_sum[i] / (i + 1)
    return cum_sum
