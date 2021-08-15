import pandas as pd
import numpy as np
from typing import List


def normalize(arr: np.ndarray,
              multi: bool = False) -> np.ndarray:
    if multi:
        return np.array([(arr[:, i] - np.min(arr[:, i])) / (np.max(arr[:, i]) - np.min(arr[:, i])) for i in
                         range(arr.shape[1])]).T
    else:
        return np.around((arr - np.min(arr)) / (np.max(arr) - np.min(arr)).T, 3)


def running_mean(arr: np.ndarray,
                 num: int) -> np.ndarray:
    cum_sum = np.cumsum(np.insert(arr=arr, values=[np.mean(arr)] * num, obj=0))
    return (cum_sum[num:] - cum_sum[:-num]) / float(num)


def cum_mean(arr: np.ndarray) -> np.ndarray:
    cum_sum = np.cumsum(arr, axis=0)
    for i in range(cum_sum.shape[0]):
        cum_sum[i] = cum_sum[i] / (i + 1)
    return cum_sum


def filter_df(data: pd.DataFrame,
              _map: str = None,
              _mode: str = None,
              _uno: int = None,
              username_dic: dict = None,
              username: str = None,
              username_lst: List[str] = None) -> pd.DataFrame:

    if _map:
        data = data[data['map'] == _map]

    if _mode:
        data = data[data['mode'] == _mode]

    if username:
        data = data[data['uno'] == username_dic[username]]

    if _uno:
        data = data[data['uno'] == _uno]

    if username_lst:
        u_lst = [username_dic[i] for i in username_lst]
        data_lst = list(data['uno'])
        data = data.iloc[[i for i, j in enumerate(data_lst) if str(j) in u_lst]]

    return data.sort_values('startDateTime', ascending=True).reset_index(drop=True)
