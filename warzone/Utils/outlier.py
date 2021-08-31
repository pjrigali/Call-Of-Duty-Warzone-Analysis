"""Various outlier detection functions.

Usage:
 ./outlier.py

Author:
 Peter Rigali - 2021-08-30
"""
from typing import List, Optional, Union
import pandas as pd
import numpy as np
from yellowbrick.regressor import CooksDistance
from statsmodels.tools import add_constant


def stack(x_arr: np.ndarray, y_arr: np.ndarray, multi: Optional[bool] = False) -> np.ndarray:
    """

    Stacks x_arr and y_arr.

    :param x_arr: An array to stack.
    :type x_arr: np.ndarray
    :param y_arr: An array to stack.
    :type y_arr: np.ndarray
    :param mutli: If True, will stack based on multiple x_arr columns, default is False. *Optional*
    :type multi: bool
    :return: Array with a x column and a y column
    :rtype: np.ndarray
    :example: *None*
    :note: *None*

    """
    lst = []
    if multi:
        for i in range((x_arr.shape[1])):
            lst.append(np.vstack([x_arr[:, i].ravel(), y_arr[:, i].ravel()]).T)
        return np.array(lst)
    else:
        lst = np.vstack([x_arr.ravel(), y_arr.ravel()]).T
    return np.where(np.isnan(lst), 0, lst)


def _cent(x_lst: List[float], y_lst: List[float]) -> List[float]:
    """

    Calculate Centroid from x and y value(s).

    :param x_lst: A list of values.
    :type x_lst: List[float]
    :param y_lst: A list of values.
    :type y_lst: List[float]
    :returns: A list of x and y values representing the centriod of two lists.
    :rtype: List[float]
    :example: *None*
    :note: *None*

    """
    return [np.sum(x_lst) / len(x_lst), np.sum(y_lst) / len(y_lst)]


def _dis(cent1: List[float], cent2: List[float]) -> float:
    """

    Calculate Distance between two centroids.

    :param cent1: An x, y coordinate representing a centroid.
    :type cent1: List[float]
    :param cent2: An x, y coordinate representing a centroid.
    :type y_lst: List[float]
    :returns: A distance measurement.
    :rtype: float
    :example: *None*
    :note: *None*

    """
    return round(np.sqrt((cent1[0] - cent2[0]) ** 2 + (cent1[1] - cent2[1]) ** 2), 4)


def outlier_std(arr: Optional[np.ndarray] = None, data: Optional[pd.DataFrame] = None, y_column: Optional[str] = None,
                _std: Optional[int] = 3, plus: Optional[bool] = True) -> np.ndarray:
    """

    Calculate Outliers using a simple std value.

    :param arr: An Array to get data from. *Optional*
    :type arr: np.ndarray
    :param data: A DataFrame to get data from. *Optional*
    :type data: pd.DataFrame
    :param y_column: A target column. *Optional*
    :type y_column: str
    :param _std: A std threshold, default is 3. *Optional*
    :type _std: int
    :param plus: If True, will grab all values above the threshold, default is True. *Optional*
    :type plus: bool
    :return: An array of indexes.
    :rtype: np.ndarray
    :example: *None*
    :note: If **arr** not passed, data and respective column names are required.

    """
    if arr is not None:
        arr = np.nan_to_num(arr)
    else:
        arr = np.nan_to_num(np.array(data[y_column].fillna(0).astype(float)))

    if plus:
        arrn = np.mean(arr, axis=0) + np.std(arr, ddof=1) * _std
    else:
        arrn = np.mean(arr, axis=0) - np.std(arr, ddof=1) * _std

    return np.where(arr <= arrn)[0]


def outlier_var(arr: Optional[np.ndarray] = None, data: Optional[pd.DataFrame] = None, y_column: Optional[str] = None,
                per: Optional[float] = 0.95, plus: Optional[bool] = True) -> np.ndarray:
    """

    Calculate Outliers using a simple var value.

    :param arr: An Array to get data from. *Optional*
    :type arr: np.ndarray
    :param data: A DataFrame to get data from. *Optional*
    :type data: pd.DataFrame
    :param y_column: A target column. *Optional*
    :type y_column: str
    :param per: A percent threshold, default is 0.95. *Optional*
    :type per: float
    :param plus: If True, will grab all values above the threshold. *Optional*
    :type plus: bool, default is True
    :return: An array of indexes.
    :rtype: np.ndarray
    :example: *None*
    :note: If **arr** not passed, data and respective column names are required.

    """
    if arr is not None:
        arr = np.nan_to_num(arr)
    else:
        arr = np.nan_to_num(np.array(data[y_column].fillna(0).astype(float)))

    temp_var = np.var(arr, ddof=1)
    dev_based = [temp_var - np.var(np.delete(arr, i), ddof=1) for i, j in enumerate(arr)]
    q = np.quantile(dev_based, per)

    if plus:
        ind = np.where(dev_based >= q)[0]
    else:
        ind = np.where(dev_based <= q)[0]

    return ind


# def outlier_regression(arr: np.ndarray = None,
#                        data: pd.DataFrame = None,
#                        x_column: str = None,
#                        y_column: str = None,
#                        _std: int = 3,
#                        plotn: bool = False) -> np.ndarray:
#
#     if arr is None:
#         x_other = np.array(data[x_column].fillna(0).astype(float))
#         y_other = np.array(data[y_column].fillna(0).astype(float))
#         arr = _stack(x_other, y_other, False)
#
#     ran = np.array(range(len(arr)))
#     mu_y = np.zeros(len(arr) - 1)
#     line_ys = []
#     for i, j in enumerate(arr):
#         xx, yy = np.delete(arr[:, 0], i), np.delete(arr[:, 1], i)
#         w1 = (np.cov(xx, yy, ddof=1) / np.var(xx, ddof=1))[0, 1]
#         new_y = w1 * ran[:-1] + (-1 * np.mean(xx) * w1 + np.mean(yy))
#
#         if plotn:
#             plt.plot(ran[:-1], new_y, alpha=0.1, color='black')
#
#         mu_y = (mu_y + new_y) / 2
#         line_ys.append(new_y)
#
#     if plotn:
#         plt.title('Test Regressions')
#         plt.plot(ran[:-1], mu_y, 'r--', label='New')
#         plt.grid(alpha=1, linestyle=(0, (3, 9)), linewidth=0.5)
#         plt.show()
#
#     reg_based = [np.mean(np.square(mu_y - j)) for i, j in enumerate(line_ys)]
#
#     if plotn:
#         w1 = np.cov(arr[:, 0], arr[:, 1], ddof=1) / np.var(arr[:, 0], ddof=1)
#         w2 = -1 * np.mean(arr[:, 0]) * w1[0, 1] + np.mean(arr[:, 1])
#         x_fit = np.linspace(np.floor(np.min(arr[:, 0])), np.ceil(np.max(arr[:, 0])), 2)
#         y_fit = w1[0, 1] * x_fit + w2
#         plt.plot(x_fit, y_fit, label='Orginal Line', color='red', alpha=0.5, linestyle='--')
#
#     # new
#     xy = np.vstack([arr[:, 0].ravel(), arr[:, 1].ravel()]).T
#     threshold = np.mean(reg_based) + np.std(reg_based, ddof=1) * _std
#
#     if plotn:
#         ind1 = np.where(reg_based <= threshold)
#         x_temp = xy[ind1][:, 0]
#         y_temp = xy[ind1][:, 1]
#         plt.scatter(x_temp, y_temp, color='orange', alpha=0.25, label='Points')
#
#     # removed
#     ind2 = np.where(reg_based >= threshold)
#     x_temp_rem = xy[ind2][:, 0]
#     y_temp_rem = xy[ind2][:, 1]
#     x2 = (x_temp_rem, y_temp_rem, ind2)
#
#     if plotn:
#         plt.scatter(x_temp_rem, y_temp_rem, color='red', label='Outliers')
#
#     # new
#     w1 = (np.cov(x_temp, y_temp) / np.var(x_temp))[0, 1]
#     w2 = -1 * np.mean(x_temp) * w1 + np.mean(y_temp)
#     new_y = w1 * x_temp + w2
#
#     if plotn:
#         plt.plot(x_temp, new_y, color='tab:blue', label='New Line')
#         plt.title('Regression Based Outlier Detection')
#         plt.xlabel('x')
#         plt.ylabel('y')
#         plt.grid(alpha=1, linestyle=(0, (3, 9)), linewidth=0.5)
#         plt.legend()
#         plt.show()
#
#     return (x1, x2)

def outlier_regression(arr: Optional[np.ndarray] = None, data: Optional[pd.DataFrame] = None,
                       x_column: Optional[str] = None, y_column: Optional[str] = None, _std: Optional[int] = 3,
                       plus: Optional[bool] = True) -> np.ndarray:
    """

    Calculate Outliers using regression.

    :param arr: An Array to get data from. *Optional*
    :type arr: np.ndarray
    :param data: A DataFrame to get data from. *Optional*
    :type data: pd.DataFrame
    :param x_column: A column for x variables. *Optional*
    :type x_column: str
    :param y_column: A column for y variables. *Optional*
    :type y_column: str
    :param _std: A std threshold, default is 3. *Optional*
    :type _std: int
    :param plus: If True, will grab all values above the threshold, default is True. *Optional*
    :type plus: bool
    :return: An array of indexes.
    :rtype: np.ndarray
    :example: *None*
    :note: If **arr** not passed, data and respective column names are required.

    """
    if arr is not None:
        arr = np.nan_to_num(arr)
    else:
        x_other = np.array(data[x_column].fillna(0).astype(float))
        y_other = np.array(data[y_column].fillna(0).astype(float))
        arr = np.nan_to_num(stack(x_other, y_other, False))

    ran = np.array(range(len(arr)))
    mu_y = np.zeros(len(arr) - 1)
    line_ys = []
    for i, j in enumerate(arr):
        xx, yy = np.delete(arr[:, 0], i), np.delete(arr[:, 1], i)
        w1 = (np.cov(xx, yy, ddof=1) / np.var(xx, ddof=1))[0, 1]
        new_y = w1 * ran[:-1] + (-1 * np.mean(xx) * w1 + np.mean(yy))
        mu_y = (mu_y + new_y) / 2
        line_ys.append(new_y)

    reg_based = [np.mean(np.square(mu_y - j)) for i, j in enumerate(line_ys)]
    threshold = np.mean(reg_based) + np.std(reg_based, ddof=1) * _std

    if plus:
        ind = np.where(reg_based >= threshold)[0]
    else:
        ind = np.where(reg_based >= threshold)[0]
    return ind


# def outlier_distance(data: np.ndarray,
#                      plotn: bool = False) -> np.ndarray:
#
#     def mahalanobis(data: np.ndarray) -> tuple:
#         x_mu = data - np.mean(data)
#         inv_covmat = np.linalg.inv(np.cov(data.T))
#         left = np.dot(x_mu, inv_covmat)
#         mahal = np.dot(left, x_mu.T).diagonal()
#         p = 1 - stats.chi2.cdf(mahal, data.shape[1] - 1)
#         return (mahal, p)
#
#     mah, p = mahalanobis(data)
#     ind = np.where(p >= .001)
#     select = np.in1d(range(data.shape[0]), ind)
#     selected = data[select]
#     un_selected = data[~select]
#     cent = tuple(np.mean(data, axis=0))
#
#     n1 = np.where(np.linalg.norm(un_selected - cent, axis=1) <= 3)
#     n2 = np.where(np.linalg.norm(selected - cent, axis=1) >= 3)
#     noise = np.concatenate((un_selected[n1], selected[n2]), axis=0)
#
#     cir = plt.Circle(cent, 1, color='k', fill=False, alpha=.75, linestyle=(0, (10, 5)), linewidth=0.5)
#     cir1 = plt.Circle(cent, 2, color='k', fill=False, alpha=.5, linestyle=(0, (7.5, 5)), linewidth=0.5)
#     cir2 = plt.Circle(cent, 3, color='k', fill=False, alpha=.25, linestyle=(0, (5, 5)), linewidth=0.5)
#
#     if plotn:
#         fig, ax = plt.subplots(figsize=(7, 7), clear=True, facecolor='white', frameon=True, dpi=100)
#         ex = 3
#         si = 10
#         lim = (np.min(data) - ex, np.max(data) + ex)
#         ax.set_xlim(lim)
#         ax.set_ylim(lim)
#         ax.grid(alpha=1, linestyle=(0, (3, 9)), linewidth=0.5)
#         ax.add_patch(cir)
#         ax.add_patch(cir1)
#         ax.add_patch(cir2)
#         ax.plot(un_selected[:, 0],
#                 un_selected[:, 1],
#                 '+',
#                 color='r',
#                 alpha=.5,
#                 markersize=si * 1,
#                 markeredgewidth=1,
#                 label='Outliers')
#         ax.plot(selected[:, 0],
#                 selected[:, 1],
#                 '.',
#                 color='tab:blue',
#                 alpha=1,
#                 markersize=si * .25,
#                 markeredgewidth=0,
#                 label='Accepted')
#         ax.plot(noise[:, 0],
#                 noise[:, 1],
#                 'v',
#                 color='skyblue',
#                 alpha=0.35,
#                 markersize=si * 1.5,
#                 markeredgewidth=0,
#                 label='Noise')
#         plt.legend()
#         plt.grid(alpha=1, linestyle=(0, (3, 9)), linewidth=0.5)
#
#     return np.concatenate((n1[0], n2[0]), axis=0)

def outlier_distance(arr: Optional[np.ndarray] = None, data: Optional[pd.DataFrame] = None,
                     x_column: Optional[str] = None, y_column: Optional[str] = None, _std: Optional[int] = 3,
                     plus: Optional[bool] = True) -> np.ndarray:
    """

    Calculate Outliers using distance measurements.

    :param arr: An Array to get data from. *Optional*
    :type arr: np.ndarray
    :param: data: A DataFrame to get data from. *Optional*
    :type data: pd.DataFrame
    :param x_column: A column for x variables. *Optional*
    :type x_column: str
    :param y_column: A column for y variables. *Optional*
    :type y_column: str
    :param _std: A std threshold, default is 3. *Optional*
    :type _std: int
    :param plus: If True, will grab all values above the threshold, default is True. *Optional*
    :type plus: bool
    :return: An array of indexes.
    :rtype: np.ndarray
    :example: *None*
    :note: If **arr** not passed, data and respective column names are required.

    """
    if arr is not None:
        arr = np.nan_to_num(arr)
    else:
        x_other = np.array(data[x_column].fillna(0).astype(float))
        y_other = np.array(data[y_column].fillna(0).astype(float))
        arr = np.nan_to_num(stack(x_other, y_other, False))

    length = len(arr)
    cent_other = _cent(arr[:, 0], arr[:, 1])
    ran = range(0, length)
    x_y_other_centers = [_dis(_cent(x_lst=[arr[i][0]], y_lst=[arr[i][1]]), cent_other) for i in ran]
    x_y_other_centers_std = np.std(x_y_other_centers, ddof=1) * _std

    if plus:
        ind = np.where(x_y_other_centers >= x_y_other_centers_std)[0]
    else:
        ind = np.where(x_y_other_centers <= x_y_other_centers_std)[0]
    return ind


# def outlier_hist(arr: np.ndarray = None,
#                  data: pd.DataFrame = None,
#                  y_column: str = None,
#                  _per: float = 0.75,
#                  plotn: bool = False) -> list:
#
#     if arr is None:
#         arr = np.array(data[y_column].fillna(0).astype(int))
#
#     length = len(arr)
#     # bins = int(np.ceil(length / np.ceil(length / 100)))
#     n, b = np.histogram(data, bins='sturges')
#     qn = np.quantile(n, _per)
#     ind = np.where(n >= qn)[0]
#     ind2 = np.where(n <= qn)[0]
#
#     bin_edges = np.array([(b[i], b[i + 1]) for i in range(len(b) - 1)])[ind]
#     bin_edges2 = np.array([(b[i], b[i + 1]) for i in range(len(b) - 1)])[ind2]
#     z_selected_ind = []
#     for i, j in enumerate(data):
#         for k, l in bin_edges:
#             if k >= j <= l:
#                 z_selected_ind.append(i)
#                 break
#
#     b1 = np.unique(bin_edges)
#     b2 = np.unique(bin_edges2)
#     select = np.in1d(data, data[z_selected_ind])
#
#     if plotn:
#         fig, ax = plt.subplots(figsize=(7, 7), clear=True, facecolor='white', frameon=True, dpi=100)
#         ax.set_title('Histogram')
#         ax.grid(alpha=.5, linestyle=(0, (3, 9)), linewidth=0.5)
#         ax.hist(data[select], bins=b1, label='Selected')
#         ax.hist(data[~select], bins=b2, label='Outliers')
#         ax.set_xlabel('Values')
#         ax.set_ylabel('Counts')
#         plt.legend()
#         plt.grid(alpha=.5, linestyle=(0, (3, 9)), linewidth=0.5)
#     return [np.where(data == i)[0][0] for i in data[np.in1d(data, data[~select])]]


def outlier_hist(arr: Optional[np.ndarray] = None, data: Optional[pd.DataFrame] = None, x_column: Optional[str] = None,
                 per: Optional[float] = 0.75, plus: Optional[bool] = True) -> np.ndarray:
    """

    Calculate Outliers using Histogram.

    :param arr: An Array to get data from. *Optional*
    :type arr: np.ndarray
    :param: data: A DataFrame to get data from. *Optional*
    :type data: pd.DataFrame
    :param x_column: A column for x variables. *Optional*
    :type x_column: str
    :param per: A std threshold, default is 3. *Optional*
    :type per: float
    :param plus: If True, will grab all values above the threshold, default is 0.75. *Optional*
    :type plus: bool
    :return: An array of indexes.
    :rtype: np.ndarray
    :example: *None*
    :note: If **arr** not passed, data and respective column names are required.

    """
    if arr is not None:
        arr = np.nan_to_num(arr)
    else:
        arr = np.nan_to_num(np.array(data[x_column].fillna(0).astype(float)))

    n, b = np.histogram(arr, bins='sturges')
    qn = np.quantile(n, per)

    if plus:
        ind = np.where(n >= qn)[0]
        bin_edges = np.array([(b[i], b[i + 1]) for i in range(len(b) - 1)])[ind]
    else:
        ind = np.where(n <= qn)[0]
        bin_edges = np.array([(b[i], b[i + 1]) for i in range(len(b) - 1)])[ind]

    z_selected_ind = []
    for i, j in enumerate(arr):
        for k, l in bin_edges:
            if k >= j <= l:
                z_selected_ind.append(i)
                break

    select = np.in1d(arr, arr[z_selected_ind])
    return np.array([np.where(arr == i)[0][0] for i in arr[np.in1d(arr, arr[~select])]])


# def outlier_knn(data,
#                _std: int = 3,
#                plotn=False
#                ):
#
#     blank = np.ones((data.shape[0], data.shape[0]))
#     for i, j in enumerate(blank):
#         blank[:, i] = np.linalg.norm(data - data[i], axis=1)
#
#     def get_knn(arr: np.ndarray,
#                 _std: int = 3,
#                 df,
#                 out=False):
#         threshold = np.mean(data) + np.std(data) * s
#         count_dic = {i: sum(data[:, i] <= threshold) for i, j in enumerate(df)}
#
#         threshold = np.floor(np.mean(list(count_dic.values())) - np.std(list(count_dic.values()), ddof=1) * _std)
#         not_selected = [i for i in count_dic.keys() if count_dic[i] <= threshold]
#         selected = [i for i in count_dic.keys() if count_dic[i] >= threshold]
#
#         select = np.in1d(range(df.shape[0]), not_selected)
#         not_s = df[select]
#         s_s = df[~select]
#
#         if out:
#             return (not_s[:, 0], not_s[:, 1])
#         else:
#             return (not_s, not_selected)
#
#     if plotn:
#         fig, ax = plt.subplots(figsize=(7, 7), clear=True, facecolor='white', frameon=True, dpi=100)
#         ex = 3
#         si = 10
#         lim = (np.min(data) - ex, np.max(data) + ex)
#         ax.set_xlim(lim)
#         ax.set_ylim(lim)
#         ax.grid(alpha=1, linestyle=(0, (3, 9)), linewidth=0.5)
#         cmap = [plt.get_cmap('plasma')(1. * i / s) for i in range(s)]
#
#         for i in list(range(s - 1, 0, -1)):
#             x1, y1, x2, y2 = getKNN(blank, i, data, True)
#             ax.plot(x1, y1, 'o', color=cmap[i], alpha=.5, markersize=si * i, markeredgewidth=1, label=str(i))
#
#         ax.plot(x2, y2, 'x', color='tab:blue', alpha=1, markersize=si * .75, markeredgewidth=.5, label='Accepted')
#         plt.legend()
#         plt.grid(alpha=1, linestyle=(0, (3, 9)), linewidth=0.5)
#
#     return getKNN(blank, s, data, out=False)

def outlier_knn(arr: Optional[np.ndarray] = None, data: Optional[pd.DataFrame] = None, x_column: Optional[str] = None,
                y_column: Optional[str] = None, _std: Optional[int] = 3, plus: Optional[bool] = True) -> np.ndarray:
    """

    Calculate Outliers using KNN.

    :param arr: An Array to get data from. *Optional*
    :type arr: np.ndarray
    :param: data: A DataFrame to get data from. *Optional*
    :type data: pd.DataFrame
    :param x_column: A column for x variables. *Optional*
    :type x_column: str
    :param y_column: A column for y variables. *Optional*
    :type y_column: str
    :param _std: A std threshold, default is 3. *Optional*
    :type _std: int
    :param plus: If True, will grab all values above the threshold, default is True. *Optional*
    :type plus: bool
    :return: An array of indexes.
    :rtype: np.ndarray
    :example: *None*
    :note: If **arr** not passed, data and respective column names are required.

    """
    if arr is not None:
        arr = np.nan_to_num(arr)
    else:
        x_other = np.array(data[x_column].fillna(0).astype(float))
        y_other = np.array(data[y_column].fillna(0).astype(float))
        arr = np.nan_to_num(stack(x_other, y_other, False))

    # threshold = np.mean(arr) + np.std(arr, ddof=1) * _std
    length = len(arr)
    ran = range(0, length)
    test_centers = [_cent([arr[ind, 0]], [arr[ind, 1]]) for ind in ran]
    distances = [_dis(cent1=i, cent2=j) for i in test_centers for j in test_centers]

    threshold = np.mean(distances) + np.std(distances, ddof=1) * _std
    if plus:
        count_dic = {i: np.sum(arr[:, i] >= threshold) for i, j in enumerate(arr)}
    else:
        count_dic = {i: np.sum(arr[:, i] <= threshold) for i, j in enumerate(arr)}

    threshold = np.floor(np.mean(list(count_dic.values())) - np.std(list(count_dic.values()), ddof=1) * _std)
    if plus:
        ind = np.where(np.array(count_dic.keys()) >= threshold)[0]
    else:
        ind = np.where(np.array(count_dic.keys()) <= threshold)[0]
    return ind


def outlier_cooks_distance(arr: Optional[np.ndarray] = None, data: Optional[pd.DataFrame] = None,
                           x_column: Optional[str] = None, y_column: Optional[str] = None, plus: Optional[bool] = True,
                           return_df: Optional[bool] = False) -> Union[np.ndarray, pd.DataFrame]:
    """

    Calculate Outliers using Cooks Distance.

    :param arr: An Array to get data from. *Optional*
    :type arr: np.ndarray
    :param data: A DataFrame to get data from. *Optional*
    :type data: pd.DataFrame
    :param x_column: A column for x variables. *Optional*
    :type x_column: str
    :param y_column: A column for y variables. *Optional*
    :type y_column: str
    :param _std: A std threshold, default is 3. *Optional*
    :type _std: int
    :param plus: If True, will grab all values above the threshold, default is True. *Optional*
    :type plus: bool
    :param return_df: If True, will return a DataFrame, default is False. *Optional*
    :type return_df: bool
    :return: An array of indexes.
    :rtype: np.ndarray or pd.DataFrame
    :example: *None*
    :note: If **arr** not passed, data and respective column names are required.

    """
    if arr is not None:
        arr = np.nan_to_num(arr)
    else:
        x_other = np.array(data[x_column].fillna(0).astype(float))
        y_other = np.array(data[y_column].fillna(0).astype(float))
        arr = np.nan_to_num(stack(x_other, y_other, False))

    result = CooksDistance().fit(add_constant(arr[:, 0]), arr[:, 1])
    distance = result.distance_

    if plus:
        ind = np.where(distance >= result.influence_threshold_)[0]
    else:
        ind = np.where(distance <= result.influence_threshold_)[0]

    if return_df:
        base_lst = [ind, distance, result.p_values_, result.influence_threshold_, result.outlier_percentage_]
        col_lst = ['Indexes', 'Distances', 'P-Values', 'Influence Threshold', 'Outlier Percentage']
        return pd.DataFrame(base_lst, columns=col_lst)
    else:
        return ind
