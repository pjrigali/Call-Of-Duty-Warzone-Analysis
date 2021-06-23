import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from yellowbrick.regressor import CooksDistance
from statsmodels.tools import add_constant


def stack(x: np.array,
          y: np.array,
          multi: bool = False,
          ) -> np.ndarray:

    lst = []
    if multi:
        for i in range((x.shape[1])):
            lst.append(np.vstack([x[:, i].ravel(), y[:, i].ravel()]).T)
        return np.array(lst)
    else:
        lst = np.vstack([x.ravel(), y.ravel()]).T
    return np.where(np.isnan(lst), 0, lst)


def outlierStd(arr: np.array,
               s: int = 3,
               plus: bool = True,
               multi: bool = False,
               ) -> tuple:
    '''
    :param arr: array of variables to be checked for outliers
    :param s: std * s, similar to a threshold
    :param plus: True will return variables above, False will return variables below
    :param multi: If the array has multible columns to be checked
    :return: tuple or list of tuples. [(data above or below theshold, (indexes below, indexes above))
    '''
    
    if multi:
        lst = []
        for i in range(arr.shape[1]):
            arrn = arr[:, i]
            if plus:
                c = np.mean(arrn) + np.std(arrn) * s
            else:
                c = np.mean(arrn) - np.std(arrn) * s
            ind = np.where(arrn <= c)[0]
            ind1 = np.where(arrn >= c)[0]
            lst.append((c, arrn[ind], (ind, ind1)))
        return tuple(lst)
    else:
        if plus:
           arrn = np.mean(arr) + np.std(arr) * s
        else:
            arrn = np.mean(arr) - np.std(arr) * s
        ind = np.where(arr <= arrn)[0]
        ind1 = np.where(arr >= arrn)[0]
        return (arrn, (ind, ind1))


def outlierDev(arr: np.array,
               per: float,
               multi: bool = False,
               ):

    if multi:
        lst = []
        for i in range(arr.shape[1]):
            arrn = arr[:, i]
            temp_var = np.var(arrn)
            dev_based = [temp_var - np.var(np.delete(arrn, i)) for i, j in enumerate(arrn)]
            q = np.quantile(dev_based, per)
            ind = np.where(dev_based >= q)[0]
            ind1 = np.where(dev_based <= q)[0]
            lst.append((q, arrn[ind], (ind, ind1)))
        return tuple(lst)
    else:
        temp_var = np.var(arr)
        dev_based = [temp_var - np.var(np.delete(arr, i)) for i, j in enumerate(arr)]
        q = np.quantile(dev_based, per)
        ind = np.where(dev_based >= q)[0]
        ind1 = np.where(dev_based <= q)[0]
        return (q, (ind, ind1))

def outlierRegression(datax,
                      datay,
                      s,
                      plotn=False
                      ):

    ran = np.array(range(len(datax)))
    mu_y = np.zeros(len(datax) - 1)
    line_ys = []
    for i, j in enumerate(datax):
        xx, yy = np.delete(datax, i), np.delete(datay, i)
        w1 = (np.cov(xx, yy) / np.var(xx))[0, 1]
        w2 = -1 * np.mean(xx) * w1 + np.mean(yy)
        new_y = w1 * ran[:-1] + w2

        if plotn:
            plt.plot(ran[:-1], new_y, alpha=0.1, color='black')

        mu_y = (mu_y + new_y) / 2
        line_ys.append(new_y)

    if plotn:
        plt.title('Test Regressions')
        plt.plot(ran[:-1], mu_y, 'r--', label='New')
        plt.grid(alpha=1, linestyle=(0, (3, 9)), linewidth=0.5)
        plt.show()

    reg_based = [np.mean(np.square(mu_y - j)) for i, j in enumerate(line_ys)]

    # Orginal
    w1 = np.cov(datax, datay) / np.var(datax)
    w2 = -1 * np.mean(datax) * w1[0, 1] + np.mean(datay)
    x_fit = np.linspace(np.floor(datax.min()), np.ceil(datax.max()), 2)
    y_fit = w1[0, 1] * x_fit + w2

    if plotn:
        plt.plot(x_fit, y_fit, label='Orginal Line', color='red', alpha=0.5, linestyle='--')

    # new
    xy = np.vstack([datax.ravel(), datay.ravel()]).T
    mu = np.mean(reg_based)
    st = np.std(reg_based) * s
    threshold = mu + st
    ind1 = np.where(reg_based <= threshold)
    x_temp = xy[ind1][:, 0]
    y_temp = xy[ind1][:, 1]

    x1 = (x_temp, y_temp, ind1)

    if plotn:
        plt.scatter(x_temp, y_temp, color='orange', alpha=0.25, label='Points')

    # removed
    ind2 = np.where(reg_based >= threshold)
    x_temp_rem = xy[ind2][:, 0]
    y_temp_rem = xy[ind2][:, 1]

    x2 = (x_temp_rem, y_temp_rem, ind2)

    if plotn:
        plt.scatter(x_temp_rem, y_temp_rem, color='red', label='Outliers')

    # new
    w1 = (np.cov(x_temp, y_temp) / np.var(x_temp))[0, 1]
    w2 = -1 * np.mean(x_temp) * w1 + np.mean(y_temp)
    new_y = w1 * x_temp + w2

    if plotn:
        plt.plot(x_temp, new_y, color='tab:blue', label='New Line')
        plt.title('Regression Based Outlier Detection')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(alpha=1, linestyle=(0, (3, 9)), linewidth=0.5)
        plt.legend()
        plt.show()

    return (x1, x2)


def outlierDistance(data,
                    plotn=False
                    ):

    def mahalanobis(data):
        x_mu = data - np.mean(data)
        inv_covmat = np.linalg.inv(np.cov(data.T))
        left = np.dot(x_mu, inv_covmat)
        mahal = np.dot(left, x_mu.T).diagonal()
        p = 1 - stats.chi2.cdf(mahal, data.shape[1] - 1)
        return (mahal, p)

    mah, p = mahalanobis(data)
    ind = np.where(p >= .001)
    select = np.in1d(range(data.shape[0]), ind)
    selected = data[select]
    un_selected = data[~select]
    cent = tuple(np.mean(data, axis=0))

    n1 = np.where(np.linalg.norm(un_selected - cent, axis=1) <= 3)
    n2 = np.where(np.linalg.norm(selected - cent, axis=1) >= 3)
    noise = np.concatenate((un_selected[n1], selected[n2]), axis=0)

    cir = plt.Circle(cent, 1, color='k', fill=False, alpha=.75, linestyle=(0, (10, 5)), linewidth=0.5)
    cir1 = plt.Circle(cent, 2, color='k', fill=False, alpha=.5, linestyle=(0, (7.5, 5)), linewidth=0.5)
    cir2 = plt.Circle(cent, 3, color='k', fill=False, alpha=.25, linestyle=(0, (5, 5)), linewidth=0.5)

    if plotn:
        fig, ax = plt.subplots(figsize=(7, 7), clear=True, facecolor='white', frameon=True, dpi=100)
        ex = 3
        si = 10
        lim = (np.min(data) - ex, np.max(data) + ex)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.grid(alpha=1, linestyle=(0, (3, 9)), linewidth=0.5)
        ax.add_patch(cir)
        ax.add_patch(cir1)
        ax.add_patch(cir2)
        ax.plot(un_selected[:, 0], un_selected[:, 1], '+', color='r', alpha=.5, markersize=si * 1,
                markeredgewidth=1, label='Outliers')
        ax.plot(selected[:, 0], selected[:, 1], '.', color='tab:blue', alpha=1, markersize=si * .25,
                markeredgewidth=0, label='Accepted')
        ax.plot(noise[:, 0], noise[:, 1], 'v', color='skyblue', alpha=0.35, markersize=si * 1.5, markeredgewidth=0,
                label='Noise')
        plt.legend()
        plt.grid(alpha=1, linestyle=(0, (3, 9)), linewidth=0.5)

    return np.concatenate((n1[0], n2[0]), axis=0)

def outlierHist(data,
                per=0.75,
                plotn=False,
                ):

    bins = int(np.ceil(len(data) / np.ceil(len(data) / 100)))
    n, b = np.histogram(data, bins)
    qn = np.quantile(n, per)
    ind = np.where(n >= qn)[0]
    ind2 = np.where(n <= qn)[0]

    bin_edges = np.array([(b[i], b[i + 1]) for i in range(len(b) - 1)])[ind]
    bin_edges2 = np.array([(b[i], b[i + 1]) for i in range(len(b) - 1)])[ind2]
    z_selected_ind = []
    for i, j in enumerate(data):
        for k, l in bin_edges:
            if k >= j <= l:
                z_selected_ind.append(i)
                break

    b1 = np.unique(bin_edges)
    b2 = np.unique(bin_edges2)
    select = np.in1d(data, data[z_selected_ind])

    if plotn:
        fig, ax = plt.subplots(figsize=(7, 7), clear=True, facecolor='white', frameon=True, dpi=100)
        ax.set_title('Histogram')
        ax.grid(alpha=.5, linestyle=(0, (3, 9)), linewidth=0.5)
        ax.hist(data[select], bins=b1, label='Selected')
        ax.hist(data[~select], bins=b2, label='Outliers')
        ax.set_xlabel('Values')
        ax.set_ylabel('Counts')
        plt.legend()
        plt.grid(alpha=.5, linestyle=(0, (3, 9)), linewidth=0.5)

    ind = [np.where(data == i)[0][0] for i in data[np.in1d(data, data[~select])]]
    return ind

def outlierKNN(data,
               s,
               plotn=False
               ):

    blank = np.ones((data.shape[0], data.shape[0]))
    for i, j in enumerate(blank):
        blank[:, i] = np.linalg.norm(data - data[i], axis=1)

    def getKNN(data, s, df, out=False):
        threshold = np.mean(data) + np.std(data) * s
        count_dic = {i: sum(data[:, i] <= threshold) for i, j in enumerate(df)}

        threshold = np.floor(np.mean(list(count_dic.values())) - np.std(list(count_dic.values())) * s)
        not_selected = [i for i in count_dic.keys() if count_dic[i] <= threshold]
        selected = [i for i in count_dic.keys() if count_dic[i] >= threshold]

        select = np.in1d(range(df.shape[0]), not_selected)
        not_s = df[select]
        s_s = df[~select]

        if out:
            return (not_s[:, 0], not_s[:, 1])
        else:
            return (not_s, not_selected)

    if plotn:
        fig, ax = plt.subplots(figsize=(7, 7), clear=True, facecolor='white', frameon=True, dpi=100)
        ex = 3
        si = 10
        lim = (np.min(data) - ex, np.max(data) + ex)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.grid(alpha=1, linestyle=(0, (3, 9)), linewidth=0.5)
        cmap = [plt.get_cmap('plasma')(1. * i / s) for i in range(s)]

        for i in list(range(s - 1, 0, -1)):
            x1, y1, x2, y2 = getKNN(blank, i, data, True)
            ax.plot(x1, y1, 'o', color=cmap[i], alpha=.5, markersize=si * i, markeredgewidth=1, label=str(i))

        ax.plot(x2, y2, 'x', color='tab:blue', alpha=1, markersize=si * .75, markeredgewidth=.5, label='Accepted')
        plt.legend()
        plt.grid(alpha=1, linestyle=(0, (3, 9)), linewidth=0.5)

    return getKNN(blank, s, data, out=False)

def outlierCD(datax,
              datay,
              plotn=False
              ):

    result = CooksDistance(show=False)

    if plotn:
        # result = cooks_distance(add_constant(datax), datay, show=True)
        result = CooksDistance(show=True).fit(add_constant(datax), datay)
    else:
        # result = cooks_distance(add_constant(datax), datay, show=False)
        result = CooksDistance(show=False).fit(add_constant(datax), datay)

    distance = result.distance_
    th = result.influence_threshold_
    ind = np.where(distance >= result.influence_threshold_)[0]
    print()
    print('Indexes ,Distances, P-Values, Influence Threshold, Outlier Percentage')
    return (ind, distance, result.p_values_, th, result.outlier_percentage_)