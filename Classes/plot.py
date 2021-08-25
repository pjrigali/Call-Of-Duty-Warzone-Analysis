import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from Utils.base import normalize, running_mean, cumulative_mean
from Classes.document_filter import DocumentFilter
from typing import List, Union, Optional
from scipy import stats
from dataclasses import dataclass
import six


def insert_every(L, char, every):
    """generates items composed of L-items interweaved with char every-so-many items"""
    for i in range(len(L)):
        yield L[i]
        if (i + 1) % every == 0:
            yield char


fonts = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']
location = ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left',  'center right', 'lower center', 'upper center', 'center']


@dataclass
class Line:
    """Creates a Line Plot"""

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
                 grid_dash_sequence: Optional[tuple] = (3, 3),
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
    """Creates a Scatter Plot"""

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
                 grid_dash_sequence: Optional[tuple] = (3, 3),
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
    """Creates a Histogram Plot"""

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
                 grid_dash_sequence: Optional[tuple] = (3, 3),
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


@dataclass
class Table:
    """Creates a Table Plot"""

    # import matplotlib
    # matplotlib.colors.to_rgba(c, alpha=None)

    def __init__(self,
                 data: pd.DataFrame,
                 label_lst: Optional[List[str]] = None,
                 fig_size: Optional[tuple] = (10, 10),
                 font_size: Optional[str] = 'medium',
                 col_widths: Optional[float] = 0.30,
                 row_colors: Optional[str] = None,
                 header_colors: Optional[str] = None,
                 edge_color: Optional[str] = 'w',
                 sequential_cells: Optional[bool] = None,
                 color_map: Optional[str] = 'Greens',
                 ):
        data['index'] = list(data.index)

        if row_colors is None:
            row_colors = ['#f1f1f2', 'w']
        if type(row_colors) is str:
            row_colors = [row_colors, 'w']
        if header_colors is None:
            header_colors = ['tab:blue', 'w']
        if type(header_colors) is str:
            header_colors = [header_colors, 'w']

        if label_lst is None:
            lst = list(data.columns)
            lst.remove('index')
            label_lst = ['index'] + lst
        data = data[label_lst]
        col_widths = [col_widths] * len(label_lst)
        colours = None
        if sequential_cells is not None:
            color_lst = []
            for col in label_lst:
                if type(data[col].iloc[0]) != str and col != 'index':
                    _norm = plt.Normalize(np.min(data[col]) - 1, np.max(data[col]) + 1)
                    temp = plt.get_cmap(color_map)(_norm(data[col]))
                elif type(data[col].iloc[0]) == str and col != 'index':
                    temp = [(1.0, 1.0, 1.0, 1.0), (0.945, 0.945, 0.949, 1.0)] * len(data)
                else:
                    temp = [(0.121, 0.466, 0.705, 0.15), (0.121, 0.466, 0.705, 0.30)] * len(data)
                temp_lst = []
                for i in range(len(data)):
                    temp_lst.append(tuple(temp[i]))
                color_lst.append(temp_lst)
            colours = np.array(pd.DataFrame(color_lst).T)

        fig, ax = plt.subplots(figsize=fig_size)
        table = ax.table(cellText=data.values, colLabels=label_lst, colWidths=col_widths, loc='center',
                         cellLoc='center', cellColours=colours)
        table.set_fontsize(font_size)

        for k, cell in six.iteritems(table._cells):
            r, c = k
            cell.set_edgecolor(edge_color)
            if r == 0:
                cell.set_text_props(weight='bold', color=header_colors[1])
                cell.set_facecolor(header_colors[0])
            else:
                if sequential_cells is None:
                    cell.set_facecolor(row_colors[r % len(row_colors)])

        ax.axis('tight')
        ax.axis('off')
        fig.tight_layout()

        self._ax = ax

    def __repr__(self):
        return 'Table Plot'

    @property
    def ax(self):
        return self._ax


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
