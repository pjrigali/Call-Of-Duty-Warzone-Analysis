.. _Classes:

Classes
*******
.. meta::
   :description: This chapter describes various classes for Analyzing and Visualizing stats.
   :keywords: Call of Duty, Warzone, Python, Data Science

This chapter documents the Classes used in this package.

.. _CallofDuty:

CallofDuty
----------
.. :currentmodule:: call_of_duty

.. class:: CallofDuty(hacker_data, squad_data, streamer_mode):

    Calculate stats for all maps/modes for each squad member.
    Loads in existing user data.

    :param user_input_dict: A dict of user inputs.
    :type user_input_dict: dict
    :param squad_data: If True, will build the Squad class, default is True.
        *Optional*
    :type squad_data: bool
    :param hacker_data: This requires a separate csv with hacker data saved.
        This data can be collected by finding hackers after the fact and
        scraping there data from CodTracker. Default is False. *Optional*
    :type hacker_data: bool
    :param streamer_mode: If True, will hide user inputted Gamertag's and Uno's,
        default is False. *Optional*
    :type streamer_mode: bool
    :example:
        .. code-block:: python

            from warzone.call_of_duty import CallofDuty
            inputs = {'repo': 'local data directory',
                      'gamertag': 'your gamertag',
                      'squad': ['friend gamertag1', 'friend gamertag2', '... etc'],
                      'file_name': 'match_data.csv'}
            cod = CallofDuty(user_input_dict=inputs,
                             squad_data=True,
                             hacker_data=False,
                             streamer_mode=False)
    :note: This will calculate and build the CallofDuty class.

.. autosummary::
    warzone.call_of_duty.CallofDuty.whole
    warzone.call_of_duty.CallofDuty.gun_dictionary
    warzone.call_of_duty.CallofDuty.last_match_date_time
    warzone.call_of_duty.CallofDuty.name_uno_dict
    warzone.call_of_duty.CallofDuty.my_uno
    warzone.call_of_duty.CallofDuty.our_df
    warzone.call_of_duty.CallofDuty.other_df
    warzone.call_of_duty.CallofDuty.hacker_df
    warzone.call_of_duty.CallofDuty.name_uno_dict_hacker
    warzone.call_of_duty.CallofDuty.user
    warzone.call_of_duty.CallofDuty.squad

.. _DocumentFilter:

DocumentFilter
--------------
DocumentFilter class object.

Uses a set of filters to return desired data to be used in later analysis.

.. :currentmodule:: document_filter

.. class:: DocumentFilter(hacker_data, squad_data, streamer_mode):

    Get a selection from a DataFrame.

    :param original_df: Input DataFrame to be filtered.
    :type original_df: pd.DataFrame
    :param map_choice: Map filter. Either 'mp_e' for Rebirth and 'mp_d' for Verdansk. *Optional*
    :type map_choice: str
    :param mode_choice: Mode filter. Either 'solo', 'duo', 'trio', or 'quad'. *Optional*
    :type mode_choice: str
    :param username: Filter by a players username. Can cause errors if same username as another player. *Optional*
    :type username: str
    :param uno: Filter by a players uno. *Optional*
    :type uno: str
    :param username_dic: Required if 'username' or 'username_lst' is used. *Optional*
    :type username_dic: dict
    :param username_lst: Filter using a list of usernames. *Optional*
    :type username_lst: List[str]
    :example:
        .. code-block:: python

            from warzone.document_filter import DocumentFilter
            doc = DocumentFilter(original_df=cod.our_df, map_choice='mp_e', mode_choice='quad')
    :note: All inputs, except **original_df**, are *Optional* and defaults are set to None.
        The example will return any data with **map = rebirth** and **mode = Quads**.
        By specifying **cod.our_df**, this will only return data related to the user and their teammates.

.. autosummary::
    warzone.document_filter.DocumentFilter.df
    warzone.document_filter.DocumentFilter.map_choice
    warzone.document_filter.DocumentFilter.mode_choice
    warzone.document_filter.DocumentFilter.uno
    warzone.document_filter.DocumentFilter.username
    warzone.document_filter.DocumentFilter.username_lst
    warzone.document_filter.DocumentFilter.unique_ids
    warzone.document_filter.DocumentFilter.ids
    warzone.document_filter.DocumentFilter.username_dic

.. _Plot:

Plot Classes
------------
Plot Class objects.

Possible Font Size Strings:
    * 'xx-small'
    * 'x-small'
    * 'small'
    * 'medium'
    * 'large'
    * 'x-large'
    * 'xx-large'

Possible Legend Locations:
    * 'best'
    * 'upper right'
    * 'upper left'
    * 'lower left'
    * 'lower right'
    * 'right'
    * 'center left'
    * 'center right'
    * 'lower center'
    * 'upper center'
    * 'center

.. :currentmodule:: plot

.. class:: Line(data):

    Class for Line plots.

    :param data: Input data.
    :type data: pd.DataFrame
    :param limit: Limit the length of data. *Optional*
    :type limit: int
    :param label_lst: List of labels to include, if None will include all columns. *Optional*
    :type label_lst: List[str]
    :param color_lst: List of colors to graph, needs to be same length as label_lst. *Optional*
    :type color_lst: List[str]
    :param normalize_x: List of columns to normalize. *Optional*
    :type normalize_x: List[str]
    :param running_mean_x: List of columns to calculate running mean. *Optional*
    :type running_mean_x: List[str]
    :param running_mean_value: Value used when calculating running mean, default = 50. *Optional*
    :type running_mean_value: int
    :param cumulative_mean_x: List of columns to calculate cumulative mean. *Optional*
    :type cumulative_mean_x: List[str]
    :param fig_size: Figure size, default = (10, 7). *Optional*
    :type fig_size: tuple
    :param ylabel: Y axis label. *Optional*
    :type ylabel: str
    :param ylabel_color: Y axis label color, default = 'black'. *Optional*
    :type ylabel_color: str
    :param ylabel_size: Y label size, default = 'medium'. *Optional*
    :type ylabel_size: str
    :param xlabel: X axis label. *Optional*
    :type xlabel: str
    :param xlabel_color: X axis label color, default = 'black'. *Optional*
    :type xlabel_color: str
    :param xlabel_size: X label size, default = 'medium'. *Optional*
    :type xlabel_size: str
    :param title: Graph title, default = 'Line Plot'. *Optional*
    :type title: str
    :param title_size: Title size, default = 'xx-large'. *Optional*
    :type title_size: str
    :param grid: If True will show grid, default = true. *Optional*
    :type grid: bool
    :param grid_alpha: Grid alpha, default = 0.75. *Optional*
    :type grid_alpha: float
    :param grid_dash_sequence: Grid dash sequence, default = (3, 3). *Optional*
    :type grid_dash_sequence: tuple
    :param grid_lineweight: Grid lineweight, default = 0.5. *Optional*
    :type grid_lineweight: float
    :param legend_fontsize: Legend fontsize, default = 'medium'. *Optional*
    :type legend_fontsize: str
    :param legend_transparency: Legend transparency, default = 0.75. *Optional*
    :type legend_transparency: float
    :param legend_location: legend location, default = 'lower right'. *Optional*
    :type legend_location: str
    :example:
        .. code-block:: python

            from warzone.plot import Line
            Line(data=data,
                 color_lst=['tab:orange', 'tab:blue'],
                 title='Weapon Preference',
                 ylabel='Percent',
                 xlabel='Date')
            plt.show()
        .. image:: https://miro.medium.com/max/700/1*qMtEJwbMB9DpOOUKx5VDtg.png
    :note: *None*

.. autosummary::
    warzone.plot.Line.ax

.. class:: Scatter(data):

    Class for Scatter plots.

    :param data: Input data.
    :type data: pd.DataFrame,
    :param limit: Limit the length of data. *Optional*
    :type limit: int
    :param label_lst: List of labels to include, if None will include all columns. *Optional*
    :type label_lst: List[str]
    :param color_lst: List of colors to graph. *Optional*
    :type color_lst: List[str]
    :param normalize_x: List of columns to normalize. *Optional*
    :type normalize_x: List[str]
    :param regression_line:  If included, requires a column str or List[str], default = None. *Optional*
    :type regression_line: List[str]
    :param regression_line_color: Color of regression line, default = 'red'. *Optional*
    :type regression_line_color: str
    :param regression_line_lineweight: Regression lineweight, default = 2.0. *Optional*
    :type regression_line_lineweight: float
    :param running_mean_x: List of columns to calculate running mean. *Optional*
    :type running_mean_x: List[str]
    :param running_mean_value: List of columns to calculate running mean. *Optional*
    :type running_mean_value: Optional[int] = 50,
    :param cumulative_mean_x: List of columns to calculate cumulative mean. *Optional*
    :type cumulative_mean_x: List[str]
    :param fig_size: default = (10, 7), *Optional*
    :type fig_size: tuple
    :param ylabel: Y axis label. *Optional*
    :type ylabel: str
    :param ylabel_color: Y axis label color, default = 'black'. *Optional*
    :type ylabel_color: str
    :param ylabel_size: Y label size, default = 'medium'. *Optional*
    :type ylabel_size: str
    :param xlabel: X axis label. *Optional*
    :type xlabel: str
    :param xlabel_color: X axis label color, default = 'black'. *Optional*
    :type xlabel_color: str
    :param xlabel_size: X label size, default = 'medium'. *Optional*
    :type xlabel_size: str
    :param title: Graph title, default = 'Scatter Plot'. *Optional*
    :type title: str
    :param title_size: Title size, default = 'xx-large'. *Optional*
    :type title_size: str
    :param grid: If True will show grid, default = true. *Optional*
    :type grid: bool
    :param grid_alpha: Grid alpha, default = 0.75. *Optional*
    :type grid_alpha: float
    :param grid_dash_sequence: Grid dash sequence, default = (3, 3). *Optional*
    :type grid_dash_sequence: tuple
    :param grid_lineweight: Grid lineweight, default = 0.5. *Optional*
    :type grid_lineweight: float
    :param legend_fontsize: Legend fontsize, default = 'medium'. *Optional*
    :type legend_fontsize: str
    :param legend_transparency: Legend transparency, default = 0.75. *Optional*
    :type legend_transparency: float
    :param legend_location: legend location, default = 'lower right'. *Optional*
    :type legend_location: str
    :param compare_two: If given will return a scatter comparing two variables, default is None. *Optional*
    :type compare_two: List[str]
    :param y_limit: If given will limit the y axis, default is None. *Optional*
    :type y_limit: float
    :example:
        .. code-block:: python

            from warzone.plot import Scatter
            Scatter(data=data,
                     compare_two=['teamSurvivalTime', 'placementPercent'],
                     normalize_x=['teamSurvivalTime'],
                     color_lst=['tab:orange'],
                     regression_line=['placementPercent'],
                     regression_line_color='tab:blue',
                     title='Team Survival Time vs Placement Percent',
                     ylabel='Placement Percent',
                     xlabel='Team Survival Time (seconds)')
             plt.show()
        .. image:: https://miro.medium.com/max/700/1*w0T6lztljOKIAFbeSR3ayQ.png
    :note: Slope of the regression line is noted in he legend.

.. autosummary::
    warzone.plot.Scatter.ax

.. class:: Histogram(data):

    Class for Histogram plots.

    :param data: Input data.
    :type data: pd.DataFrame,
    :param limit: Limit the length of data. *Optional*
    :type limit: int
    :param label_lst: List of labels to include, if None will include all columns. *Optional*
    :type label_lst: List[str]
    :param color_lst: List of colors to graph. *Optional*
    :type color_lst: List[str]
    :param include_norm: Include norm. If included, requires a column str, default = None. *Optional*
    :type include_norm: str
    :param norm_color: Norm color, default = 'red'. *Optional*
    :type norm_color: str
    :param norm_lineweight: Norm lineweight, default = 1.0. *Optional*
    :type norm_lineweight: float
    :param norm_ylabel: Norm Y axis label. *Optional*
    :type norm_ylabel: str
    :param norm_legend_location: Location of norm legend, default = 'upper right'. *Optional*
    :type norm_legend_location: str
    :param fig_size: default = (10, 7), *Optional*
    :type fig_size: tuple
    :param bins: Way of calculating bins, default = 'sturges'. *Optional*
    :type bins: str
    :param hist_type: Type of histogram, default = 'bar'. *Optional*
    :type hist_type: str
    :param stacked: If True, will stack histograms, default = False. *Optional*
    :type stacked: bool
    :param ylabel: Y axis label. *Optional*
    :type ylabel: str
    :param ylabel_color: Y axis label color, default = 'black'. *Optional*
    :type ylabel_color: str
    :param ylabel_size: Y label size, default = 'medium'. *Optional*
    :type ylabel_size: str
    :param ytick_rotation:
    :type ytick_rotation: Optional[int] = 0,
    :param xlabel: X axis label. *Optional*
    :type xlabel: str
    :param xlabel_color: X axis label color, default = 'black'. *Optional*
    :type xlabel_color: str
    :param xlabel_size: X label size, default = 'medium'. *Optional*
    :type xlabel_size: str
    :param xtick_rotation:
    :type xtick_rotation: Optional[int] = 0,
    :param title: Graph title, default = 'Histogram'. *Optional*
    :type title: str
    :param title_size: Title size, default = 'xx-large'. *Optional*
    :type title_size: str
    :param grid: If True will show grid, default = true. *Optional*
    :type grid: bool
    :param grid_alpha: Grid alpha, default = 0.75. *Optional*
    :type grid_alpha: float
    :param grid_dash_sequence: Grid dash sequence, default = (3, 3). *Optional*
    :type grid_dash_sequence: tuple
    :param grid_lineweight: Grid lineweight, default = 0.5. *Optional*
    :type grid_lineweight: float
    :param legend_fontsize: Legend fontsize, default = 'medium'. *Optional*
    :type legend_fontsize: str
    :param legend_transparency: Legend transparency, default = 0.75. *Optional*
    :type legend_transparency: float
    :param legend_location: legend location, default = 'lower right'. *Optional*
    :type legend_location: str
    :example:
        .. code-block:: python

            from warzone.plot import Histogram
            Histogram(data=data,
                      label_lst=['kills_log'],
                      include_norm='kills_log',
                      title='Kills Histogram')
            plt.show()
        .. image:: https://miro.medium.com/max/700/1*gzO4N258m-0pEb-5pmaKFA.png
    :note: *None*

.. autosummary::
    warzone.plot.Histogram.ax

.. class:: Table(data):

    Class for Table plots.

    `Possible Color Maps <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_

    :param data: Input data.
    :type data: pd.DataFrame
    :param label_lst: List of labels to include, if None will include all columns. *Optional*
    :type label_lst: List[str]
    :param fig_size: default = (10, 10), *Optional*
    :type fig_size: tuple
    :param font_size: Font size inside cells, default = 'medium'. *Optional*
    :type font_size: str
    :param font_color: Color of text inside cells, default is 'black'. *Optional*
    :type font_color: str
    :param col_widths: Width of columns, default = 0.30. *Optional*
    :type col_widths: float
    :param row_colors: Color of rows. *Optional*
    :type row_colors: str
    :param header_colors: Header of table color. *Optional*
    :type header_colors: str
    :param edge_color: Color of cell edges, default = 'w'. *Optional*
    :type edge_color: str
    :param sequential_cells: If True will color ever other row. *Optional*
    :type sequential_cells: bool
    :param color_map: Color map used in cells, default = 'Greens'. *Optional*
    :type color_map: str
    :example:
        .. code-block:: python

            from warzone.plot import Table
            Table(data=data,
                  col_widths=0.15,
                  fig_size=(10, 4),
                  sequential_cells=True)
            plt.show()
        .. image:: https://cdn-images-1.medium.com/max/800/1*AE_sEF5gWDrtUaPHogR7CQ.png

        Or with color: (color_map = "Oranges")

        .. image:: https://miro.medium.com/max/700/1*WIh5zrwCc5pZRJJVS6WMeQ.png
    :note: Will have to update **figure_size** and **col_widths** depending on the size of the table.
        If a cmap is provided, only float dtype columns will show changes.

.. autosummary::
    warzone.plot.Table.ax

.. _Regression:

Regression
----------
Regression class object.

.. :currentmodule:: regression

.. class:: Regression(doc_filter, x_column, y_column):

    Class of calculating a linear regression.

    :param doc_filter: Input DocumentFilter.
    :type doc_filter: DocumentFilter
    :param x_column: Name of column or columns to be used in regression analysis.
    :type x_column: str, or List[str]
    :param y_column: Name of column to be used as y variable in regression.
    :type y_column: str
    :example:
        .. code-block:: python

            from warzone.document_filter import DocumentFilter
            from warzone.regression import Regression
            doc = DocumentFilter(original_df=cod.our_df, map_choice='mp_e', mode_choice='quad')
            model = Regression(doc_filter=doc, x_column='kills', y_column='placementPercent')
    :note: This will return a Regression object with regression result information.

.. autosummary::
    warzone.regression.Regression.r2
    warzone.regression.Regression.constant_coefficient
    warzone.regression.Regression.x_coefficient
    warzone.regression.Regression.lower_confidence
    warzone.regression.Regression.upper_confidence
    warzone.regression.Regression.pvalue
    warzone.regression.Regression.residuals
    warzone.regression.Regression.mse
    warzone.regression.Regression.ssr
    warzone.regression.Regression.ess
    warzone.regression.Regression.confidence
    warzone.regression.Regression.coefficients

.. _User:

User
----
User class object.

.. :currentmodule:: user

.. class:: User:

    Organizes the Users input data.

    :param info: User input dict.
    :type info: dict
    :example:
        .. code-block:: python

            from warzone.user import User
            inputs = {'repo': 'local data directory',
                      'gamertag': 'your gamertag',
                      'squad': ['friend gamertag1', 'friend gamertag2', '... etc'],
                      'file_name': 'match_data.csv'}
            user = User(info=inputs)
    :note: *This class is not intended to be used outside of creating the CallofDuty Class.*

.. autosummary::
    warzone.user.User.file_name
    warzone.user.User.repo
    warzone.user.User.gamertag
    warzone.user.User.squad_lst

.. _Squad:

Squad
-----
Squad class object.

.. :currentmodule:: squad

.. class:: Performance:

    The Performance class is used to evaluate a players performance on a given map and mode

    :param original_df: Input data.
    :type original_df: pd.DataFrame
    :param nap_choice: Map filter. Either 'mp_e' for Rebirth and 'mp_d' for Verdansk.
    :type map_choice: str
    :param mode_choice: Mode filter. Either 'solo', 'duo', 'trio', or 'quad'.
    :type mode_choice: str
    :param uno: Input person uno Id.
    :type uno: str
    :example: *None*
    :note: *This class is not intended to be used outside of creating the Squad Class.*

.. autosummary::
    warzone.squad.Performance.map
    warzone.squad.Performance.mode
    warzone.squad.Performance.stats

.. class:: Person:

    The Person class is used to gather all map/mode stats for a given player

    :param original_df: Input data.
    :type original_df: pd.DataFrame
    :param uno: Input person uno Id.
    :type uno: str
    :param gamertag: Input person's gamertag.
    :type gamertag: str
    :example: *None*
    :note: *This class is not intended to be used outside of creating the Squad Class.*

.. autosummary::
    warzone.squad.Person.gamertag
    warzone.squad.Person.uno
    warzone.squad.Person.rebirth
    warzone.squad.Person.verdansk

.. class:: Squad(squad_lst, original_df, uno_name_dic):

    Calculate stats for all maps/modes for each squad member.

    :param squad_lst: List of gamertags. Include your gamertag in the list.
    :type squad_lst: List[str]
    :param original_df: Original DataFrame for stats to be calculated from.
    :type original_df: pd.DataFrame
    :param uno_name_dic: A dict of all gamertags and respective unos.
    :type uno_name_dic: dict
    :example:
        .. code-block:: python

            from warzone.user import User
            from warzone.squad import Squad
            _User = User(info=user_inputs)
            _Squad = Squad(squad_lst=_User.squad_lst,
                           original_df=cod.our_df,
                           uno_name_dic=cod.name_uno_dict)
    :note: This will calculate and return the stats for all squad members.
        This is not intended to be used outside of building the CallofDuty Class.

.. autosummary::
    warzone.squad.Squad.squad_dic
    warzone.squad.Squad.squad_df
