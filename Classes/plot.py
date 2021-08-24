import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from Utils.base import normalize, running_mean, cumulative_mean
from Classes.document_filter import DocumentFilter
from typing import List, Union, Optional
from scipy import stats
from dataclasses import dataclass


def insert_every(L, char, every):
    '''generates items composed of L-items interweaved with char every-so-many items'''
    for i in range(len(L)):
        yield L[i]
        if (i + 1) % every == 0:
            yield char


fonts = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']
location = ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left',  'center right', 'lower center', 'upper center', 'center']


@dataclass
class Line:

    def __init__(self,
                 data: pd.DataFrame,
                 limit: Optional[int] = None,
                 label_lst: Optional[List[str]] = None,
                 color_lst: Optional[List[str]] = None,
                 normalize_x: Optional[List[str]] = None,
                 running_mean_x: Optional[List[str]] = None,
                 running_mean_value: Optional[int] = 50,
                 cumulative_mean_x: Optional[List[str]] = None,
                 fig_size: Optional[tuple] = (10, 7),
                 ylabel: Optional[str] = None,
                 ylabel_color: Optional[str] = 'black',
                 ylabel_size: Optional[str] = 'medium',
                 xlabel: Optional[str] = None,
                 xlabel_color: Optional[str] = 'black',
                 xlabel_size: Optional[str] = 'medium',
                 title: Optional[str] = 'Line Plot',
                 title_size: Optional[str] = 'xx-large',
                 grid: Optional[bool] = True,
                 grid_alpha: Optional[float] = 0.75,
                 grid_dash_sequence: Optional[tuple] = (1, 3),
                 grid_lineweight: Optional[float] = 0.5,
                 legend_fontsize: Optional[str] = 'medium',
                 legend_transparency: Optional[float] = 0.75,
                 legend_location: Optional[str] = 'lower right',
                 ):

        if label_lst is None:
            label_lst = list(data.columns)

        if color_lst is None:
            n = len(label_lst)
            if n == 1:
                color_lst = ['tab:orange']
            else:
                color_lst = [plt.get_cmap('viridis')(1. * i / n) for i in range(n)]

        if normalize_x is None:
            normalize_x = []

        if running_mean_x is None:
            running_mean_x = []

        if cumulative_mean_x is None:
            cumulative_mean_x = []

        fig, ax = plt.subplots(figsize=fig_size)

        if limit:
            data = data[:limit]

        count = 0
        for ind in label_lst:

            d = data[ind]
            if ind in normalize_x:
                d = normalize(d)
            elif ind in running_mean_x:
                d = running_mean(d, running_mean_value)
            elif ind in cumulative_mean_x:
                d = cumulative_mean(d)

            ax.plot(d, color=color_lst[count], label=ind)
            count += 1

        ax.set_ylabel(ylabel, color=ylabel_color, fontsize=ylabel_size)
        ax.tick_params(axis='y', labelcolor=ylabel_color)
        ax.set_title(title, fontsize=title_size)

        if grid:
            ax.grid(alpha=grid_alpha, linestyle=(0, grid_dash_sequence), linewidth=grid_lineweight)

        ax.set_xlabel(xlabel, color=xlabel_color, fontsize=xlabel_size)
        ax.legend(fontsize=legend_fontsize, framealpha=legend_transparency, loc=legend_location, frameon=True)
        self._ax = ax

    def __repr__(self):
        return 'Line Plot'

    @property
    def ax(self):
        return self._ax


@dataclass
class Scatter:

    def __init__(self,
                 data: pd.DataFrame,
                 limit: Optional[int] = None,
                 label_lst: Optional[List[str]] = None,
                 color_lst: Optional[List[str]] = None,
                 normalize_x: Optional[List[str]] = None,
                 regression_line: Optional[List[str]] = None,
                 regression_line_color: Optional[str] = 'r',
                 regression_line_lineweight: Optional[float] = 2.0,
                 running_mean_x: Optional[List[str]] = None,
                 running_mean_value: Optional[int] = 50,
                 cumulative_mean_x: Optional[List[str]] = None,
                 fig_size: Optional[tuple] = (10, 7),
                 ylabel: Optional[str] = None,
                 ylabel_color: Optional[str] = 'black',
                 ylabel_size: Optional[str] = 'medium',
                 xlabel: Optional[str] = None,
                 xlabel_color: Optional[str] = 'black',
                 xlabel_size: Optional[str] = 'medium',
                 title: Optional[str] = 'Scatter Plot',
                 title_size: Optional[str] = 'xx-large',
                 grid: Optional[bool] = True,
                 grid_alpha: Optional[float] = 0.75,
                 grid_dash_sequence: Optional[tuple] = (1, 3),
                 grid_lineweight: Optional[float] = 0.5,
                 legend_fontsize: Optional[str] = 'medium',
                 legend_transparency: Optional[float] = 0.75,
                 legend_location: Optional[str] = 'lower right',
                 ):

        if label_lst is None:
            label_lst = list(data.columns)

        if color_lst is None:
            n = len(label_lst)
            if n == 1:
                color_lst = ['tab:orange']
                regression_line_color = 'tab:blue'
            else:
                color_lst = [plt.get_cmap('viridis')(1. * i / n) for i in range(n)]

        if normalize_x is None:
            normalize_x = []

        if running_mean_x is None:
            running_mean_x = []

        if cumulative_mean_x is None:
            cumulative_mean_x = []

        if regression_line is None:
            regression_line = []

        fig, ax = plt.subplots(figsize=fig_size)

        if limit:
            data = data[:limit]

        x_axis = range(len(data))
        count = 0
        for ind in label_lst:

            d = data[ind]
            if ind in normalize_x:
                d = normalize(d)
            elif ind in running_mean_x:
                d = running_mean(d, running_mean_value)
            elif ind in cumulative_mean_x:
                d = cumulative_mean(d)

            ax.scatter(x=x_axis, y=d, color=color_lst[count], label=ind)

            if ind in regression_line:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_axis, d)

                if len(label_lst) == 1:
                    c = regression_line_color
                else:
                    c = color_lst[count]

                ax.plot(x_axis, slope * x_axis + intercept, color=c, label=ind+'_ols_'+str(round(slope, 2)),
                        linestyle='--', linewidth=regression_line_lineweight)
            count += 1

        ax.set_ylabel(ylabel, color=ylabel_color, fontsize=ylabel_size)
        ax.tick_params(axis='y', labelcolor=ylabel_color)
        ax.set_title(title, fontsize=title_size)

        if grid:
            ax.grid(alpha=grid_alpha, linestyle=(0, grid_dash_sequence), linewidth=grid_lineweight)

        ax.set_xlabel(xlabel, color=xlabel_color, fontsize=xlabel_size)
        ax.legend(fontsize=legend_fontsize, framealpha=legend_transparency, loc=legend_location, frameon=True)
        self._ax = ax

    def __repr__(self):
        return 'Scatter Plot'

    @property
    def ax(self):
        return self._ax


@dataclass
class Histogram:

    def __init__(self,
                 data: pd.DataFrame,
                 color_lst: Optional[List[str]] = None,
                 label_lst: Optional[List[str]] = None,
                 limit: Optional[int] = None,
                 include_norm: Optional[str] = None,
                 norm_color: Optional[str] = 'r',
                 norm_lineweight: Optional[float] = 1.0,
                 norm_ylabel: Optional[str] = None,
                 norm_legend_location: Optional[str] = 'upper right',
                 fig_size: Optional[tuple] = (10, 7),
                 bins: Optional[str] = 'sturges',
                 hist_type: Optional[str] = 'bar',
                 stacked: Optional[bool] = False,
                 ylabel: Optional[str] = None,
                 ylabel_color: Optional[str] = 'black',
                 ylabel_size: Optional[str] = 'medium',
                 ytick_rotation: Optional[int] = 0,
                 xlabel: Optional[str] = None,
                 xlabel_color: Optional[str] = 'black',
                 xlabel_size: Optional[str] = 'medium',
                 xtick_rotation: Optional[int] = 0,
                 title: Optional[str] = 'Histogram',
                 title_size: Optional[str] = 'xx-large',
                 grid: Optional[bool] = True,
                 grid_alpha: Optional[float] = 0.75,
                 grid_dash_sequence: Optional[tuple] = (1, 3),
                 grid_lineweight: Optional[float] = 0.5,
                 legend_fontsize: Optional[str] = 'medium',
                 legend_transparency: Optional[float] = 0.75,
                 legend_location: Optional[str] = 'lower right',
                 ):

        if label_lst is None:
            label_lst = list(data.columns)

        if color_lst is None:
            n = len(label_lst)
            if n == 1:
                color_lst = ['tab:orange']
                norm_color = 'tab:blue'
            else:
                color_lst = [plt.get_cmap('viridis')(1. * i / n) for i in range(n)]

        fig, ax = plt.subplots(figsize=fig_size)

        if limit:
            data = data[:limit]

        count = 0
        for ind in label_lst:
            ax.hist(data[ind], bins=bins, color=color_lst[count], label=ind, stacked=stacked, histtype=hist_type)
            count += 1

        ax.set_ylabel(ylabel, color=ylabel_color, fontsize=ylabel_size)
        ax.tick_params(axis='y', labelcolor=ylabel_color, rotation=ytick_rotation)
        ax.set_title(title, fontsize=title_size)

        if grid:
            ax.grid(alpha=grid_alpha, linestyle=(0, grid_dash_sequence), linewidth=grid_lineweight)

        ax.set_xlabel(xlabel, color=xlabel_color, fontsize=xlabel_size)
        ax.tick_params(axis='x', labelcolor=ylabel_color, rotation=xtick_rotation)
        ax.legend(fontsize=legend_fontsize, framealpha=legend_transparency, loc=legend_location, frameon=True)

        ax1 = None
        if include_norm:
            d = data[include_norm]
            norm_data = np.random.normal(np.mean(d), np.std(d, ddof=1), len(d))
            _mu, _std = norm.fit(norm_data)
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            ax1 = ax.twinx()
            ax1.plot(x, norm.pdf(x, _mu, _std), color=norm_color, linewidth=norm_lineweight, linestyle='--',
                     label="Fit Values: mu {:.2f} and std {:.2f}".format(_mu, _std))
            ax1.set_ylabel(norm_ylabel, color=norm_color)
            ax1.tick_params(axis='y', labelcolor=norm_color)
            ax1.legend(fontsize=legend_fontsize, framealpha=legend_transparency, loc=norm_legend_location, frameon=True)
        self._ax = ax
        self._ax1 = ax1

    def __repr__(self):
        return 'Histogram Plot'

    @property
    def ax(self):
        return self._ax, self._ax1


# @dataclass
# class BarChart:
#
#     def __init__(self,
#                  data: pd.DataFrame,
#                  label_lst: Optional[List[str]] = None,
#                  xlabel: Optional[str] = 'X Axis',
#                  xlabel_size: Optional[str] = 'medium',
#                  ylabel: Optional[str] = 'Y Axis',
#                  ylabel_size: Optional[str] = 'medium',
#                  title: Optional[str] = 'Bar Chart',
#                  title_size: Optional[str] = 'xx-large',
#                  limit: int = None,
#                  spacing: int = None,
#                  include_mu: bool = None,
#                  mu_color: Optional[str] = 'r',
#                  color_lst: list = None,
#                  xtick_rotation: int = -90,
#                  xtick_size: Optional[str] = 'small',
#                  grid: bool = True,
#                  grid_alpha: float = 0.75,
#                  grid_lineweight: float = 0.5,
#                  grid_dash_sequence: tuple = (1, 3),
#                  legend_transparency: float = 0.5,
#                  legend_fontsize: Optional[str] = 'medium',
#                  legend_location: Optional[str] = 'upper right',
#                  fig_size: Optional[tuple] = (10, 7)):
#
#         if label_lst is None:
#             label_lst = list(data.columns)
#
#         if color_lst is None:
#             n = len(label_lst)
#             color_lst = [plt.get_cmap('viridis')(1. * i / n) for i in range(n)]
#
#         if include_mu is not None:
#             mu = int(np.ceil(np.mean(y)))
#
#         if limit is not None:
#             data = data[:limit]
#
#         fig, ax = plt.subplots(figsize=fig_size)
#
#         count = 0
#         for ind in label_lst:
#             y = data[ind]
#             x_axis = range(len(y))
#             if spacing is not None:
#                 x_axis = range(len([i for i in insert_every(x_axis, '', spacing)]))
#                 y = [i for i in insert_every(y, 0, spacing)]
#             ax.bar(x_axis, y, align='center', color=color_lst[count], label=ind)
#             count += 1
#
#         if include_mu is not None:
#             ax.plot(x_axis, [mu] * len(x), linestyle='--', color=mu_color, label='mu: ' + str(mu))
#
#         plt.xticks(x_axis, x, rotation=xtick_rotation, fontsize=xtick_size)
#         plt.ylabel(ylabel, fontsize=ylabel_size)
#         plt.xlabel(xlabel, fontsize=xlabel_size)
#         plt.title(title, fontsize=title_size)
#
#         if grid:
#             ax.grid(alpha=grid_alpha, linestyle=(0, grid_dash_sequence), linewidth=grid_lineweight)
#
#         if include_mu is not None:
#             plt.legend(fontsize=legend_fontsize, framealpha=legend_transparency, loc=legend_location, frameon=True)
#         plt.show()
#
#     def __repr__(self):
#         return 'Bar Chart Plot'


# @dataclass
# class Plot:
#
#     def __init__(self,
#                  doc_filter: DocumentFilter,
#                  col_lst: Union[str, List[str]],
#                  line: Optional[bool] = False,
#                  scat: Optional[bool] = False,
#                  histo: Optional[bool] = False,
#                  bar: Optional[bool] = False,
#                  ):
#
#         self._data = doc_filter.df[col_lst]
#         self._col_lst = col_lst
#         self._line = line
#         self._scat = scat
#         self._histo = histo
#         self._bar = bar
#
#         if self._line:
#             Line(data=self._data, label_lst=self._col_lst)
#         if self._scat:
#             Scatter(data=self._data, label_lst=self._col_lst)
#         if self._histo:
#             Histogram(data=self._data)
#         if self._bar:
#             BarChart(data=self._data)
#
#     def __repr__(self):
#         return 'Plot'
