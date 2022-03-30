"""Various calculations preformed on arrays.

Usage:
 ./warzone/utils/base.py

Author:
 Peter Rigali - 2021-08-30
"""
from typing import Optional, Union, List
import numpy as np
import pandas as pd
from pyjr.utils.tools import _to_metatype


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


def to_type(data: Union[list, np.float64, np.float32, np.float16, np.float_, np.int64, np.int32, np.int16, np.int8,
                        np.int_, float, int], new_type: str) -> Union[List[int], List[float], int, float]:
    """Converts objects to a set item"""
    if new_type == 'int':
        if type(data) == list:
            return [int(i) for i in data]
        else:
            return int(data)
    elif new_type == 'float':
        if type(data) == list:
            return [float(i) for i in data]
        else:
            return float(data)
    else:
        raise AttributeError('new_type can be "int" or "float.')


def to_list(data: Union[list, np.ndarray, pd.Series, int, float]) -> Union[List[int], List[float], float, int]:
    """Converts list adjacent objects to a list and passes int/float objects"""
    if type(data) == list:
        return data
    elif type(data) in [np.ndarray, pd.Series]:
        return data.tolist()
    elif type(data) in [int, float]:
        return data
    else:
        raise AttributeError('data needs to have a type of {np.ndarray, pd.Series, list}')


def remove_nan(data: list, replace_val: Optional[Union[int, float, str]] = None,
               keep_nan: Optional[bool] = False) -> list:
    """Remove or replace nan values"""
    if replace_val:
        if replace_val == 'mean':
            replace_val = native_mean(data=remove_nan(data=data))
        elif type(replace_val) in [int, float]:
            pass
        else:
            raise AttributeError('replace_val needs to be an int or float. If "mean" is passed, will use mean.')
        return [i if i == i and i is not None else replace_val for i in data]
    if keep_nan is False:
        return [i for i in data if i == i and i is not None]
    else:
        return [i if i == i and i is not None else None for i in data]


def native_mean(data: Union[list, np.ndarray, pd.Series]) -> float:
    """

    Calculate Mean of a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :return: Returns the mean.
    :rtype: float
    :example: *None*
    :note: *None*

    """
    data = remove_nan(data=to_list(data=data))
    if len(data) != 0:
        return sum(data) / len(data)
    else:
        return 0.0


def round_to(
        data: Union[list, np.ndarray, pd.Series, np.float64, np.float32, np.float16, np.float_, np.int64, np.int32,
                    np.int16, np.int8, np.int_, float, int], val: Union[int, float],
        remainder: Optional[bool] = False) -> Union[List[float], float]:
    """

    Rounds an np.array, pd.Series, or list of values to the nearest value.

    :param data: Input data.
    :type data: list, np.ndarray, pd.Series, int, float, or any of the numpy int/float variations
    :param val: Value to round to. If decimal, will be that number divided by.
    :type val: int
    :param remainder: If True, will round the decimal, default is False. *Optional*
    :type remainder: bool
    :return: Rounded number.
    :rtype: List[float] or float
    :example:
        >>> # With remainder set to True.
        >>> lst = [4.3, 5.6]
        >>> round_to(data=lst, val=4, remainder=True) # [4.25, 5.5]
        >>>
        >>>  # With remainder set to False.
        >>> lst = [4.3, 5.6]
        >>> round_to(data=lst, val=4, remainder=False) # [4, 4]
        >>>
    :note: Single int or float values can be passed.

    """
    if type(val) == int:
        val = float(val)

    if type(data) not in [list, pd.Series, np.ndarray]:
        data = to_type(data=data, new_type='float')
        if remainder is True:
            return round(data * val) / val
        else:
            return round(data / val) * val
    else:
        data = to_type(data=remove_nan(data=to_list(data=data), replace_val=0), new_type='float')
        if remainder is True:
            return [round(item * val) / val for item in data]
        else:
            return [round(item / val) * val for item in data]


def calc_gini(data: Union[list, np.ndarray, pd.Series]) -> float:
    """

    Calculate the Gini Coef for a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :return: Gini value.
    :rtype: float
    :example:
        >>> lst = [4.3, 5.6]
        >>> calc_gini(data=lst, val=4, remainder=True) # 0.05445544554455435
    :note: The larger the gini coef, the more consolidated the chips on the table are to one person.

    """
    data = to_list(data=data)
    sorted_list = sorted(data)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(data) / 2.
    return (fair_area - area) / fair_area


def slc(d: pd.DataFrame, c: str, v: Union[float, int, str, object]) -> pd.DataFrame:
    """
    Select function filters a dataframe using col and a value.

    :param d: Selected DataFrame
    :type d: pd.DataFrame
    :param c: Column
    :type c: str
    :param v: Value to look for.
    :type v: float, int, str, or object
    :return: Returns part of DataFrame matching value within desired column.
    :rtype: pd.DataFrame


    """
    return d[d[c] == v]


def ilc(d: pd.DataFrame, i) -> pd.DataFrame:
    """
    iloc function filters a dataframe using a list of values.

    :param d: Selected DataFrame
    :type d: pd.DataFrame
    :param i: list of indexes to filter rows by.
    :type i: List like objects
    :return: Returns part of DataFrame matching the input index list.
    :rtype: pd.DataFrame


    """
    return d.iloc[_to_metatype(data=i, dtype='list')]
