.. _Classes:

Classes
*******
.. meta::
   :description: This chapter describes various classes for Analyzing and Visualizing stats.
   :keywords: Call of Duty, Warzone, Python, Data Science

This chapter documents the classes used.

.. _CallofDuty:

CallofDuty
----------
.. :currentmodule:: call_of_duty

.. class:: CallofDuty(hacker_data, squad_data, streamer_mode):

    Calculate stats for all maps/modes for each squad member.

    :param user_input_dict: A dict of user inputs.
    :type user_input_dict: dict
    :param squad_data: If True, will build the Squad class. default is True. *Optional*
    :type squad_data: bool
    :param hacker_data: This Requires a seperate csv with hacker data saved. This data can be collected by
        finding hackers after the fact and scraping there data from CodTracker, this can then be used to find
        hackers in other games. Default is False. *Optional*
    :type hacker_data: bool
    :param streamer_mode: If True, will hide User inputted Gamertag's and Uno's. default is False. *Optional*
    :type streamer_mode: bool
    :example:
        >>> from call_of_duty import CallofDuty
        >>> inputs = {'repo': 'local data directory',
        >>>        'gamertag': 'your gamertag',
        >>>        'squad': ['friend gamertag1', 'friend gamertag2', '... etc'],
        >>>        'file_name': 'match_data.csv'}
        >>> cod = CallofDuty(user_input_dict=inputs, squad_data=True, hacker_data=False, streamer_mode=False)
    :note: This will calculate and build the CallofDuty class.

.. autosummary::
    call_of_duty.CallofDuty.whole
    call_of_duty.CallofDuty.gun_dictionary
    call_of_duty.CallofDuty.last_match_date_time
    call_of_duty.CallofDuty.name_uno_dict
    call_of_duty.CallofDuty.my_uno
    call_of_duty.CallofDuty.our_df
    call_of_duty.CallofDuty.other_df
    call_of_duty.CallofDuty.hacker_df
    call_of_duty.CallofDuty.name_uno_dict_hacker
    call_of_duty.CallofDuty.user
    call_of_duty.CallofDuty.squad

.. _DocumentFilter:

DocumentFilter
--------------
DocumentFilter class objects.

.. :currentmodule:: document_filter

.. class:: DocumentFilter(hacker_data, squad_data, streamer_mode):

    Get a selection from a DataFrame.
    Uses a set of filters to return a desired set of data to be used in later analysis.

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
    :param username_dic: Required if 'username' or 'username_lst' is used. {username1: uno1, username2: uno2, etc}. *Optional*
    :type username_dic: dict
    :param username_lst: Filter using a list of usernames. *Optional*
    :type username_lst: List[str]
    :example:
        >>> from document_filter import DocumentFilter
        >>> doc = DocumentFilter(original_df=cod.our_df, map_choice='mp_e', mode_choice='quad')
    :note: All inputs, except original_df,  are *Optional* amd defaults are set to None.
        This will return any data with map = rebirth and mode = Quads.
        By specifiying 'cod.our_df', this will only return data related to the user.

.. autosummary::
    document_filter.DocumentFilter.df
    document_filter.DocumentFilter.map_choice
    document_filter.DocumentFilter.mode_choice
    document_filter.DocumentFilter.uno
    document_filter.DocumentFilter.username
    document_filter.DocumentFilter.username_lst
    document_filter.DocumentFilter.unique_ids
    document_filter.DocumentFilter.ids
    document_filter.DocumentFilter.username_dic

.. _Plot:

Plot
----
Plot class objects.

font size = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']

Legend location = ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center']

.. :currentmodule:: plot

.. class:: Line:

    Class for plotting line plots.

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
    :example: *None*
    :note: *None*

.. autosummary::
    plot.Line.ax

.. class:: Scatter:

    Class for plotting scatter plots.

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
    :param compare_two: If given will return a scatter comparing two variables,default is None. *Optional*
    :type compare_two: List[str]
    :param y_limit: If given will limit the y axis.
    :type y_limit: float
    :example: *None*
    :note: *None*

.. autosummary::
    plot.Scatter.ax

.. class:: Histogram:

    Class for plotting histograms.

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
    :example: *None*
    :note: *None*

.. autosummary::
    plot.Histogram.ax

.. class:: Table:

    Class for plotting tables.

    :param data: Input data.
    :type data: pd.DataFrame
    :param label_lst: List of labels to include, if None will include all columns. *Optional*
    :type label_lst: List[str]
    :param fig_size: default = (10, 10), *Optional*
    :type fig_size: tuple
    :param font_size: Font size inside cells, default = 'medium'. *Optional*
    :type font_size: str
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
    :example: *None*
    :note: *None*

.. autosummary::
    plot.Table.ax

.. _Regression:

Regression
----------
Regression class object.

.. :currentmodule:: regression

.. class:: Regression:

    Calculate a linear regression.

    :param doc_filter: Input DocumentFilter.
    :type doc_filter: DocumentFilter
    :param x_column: Name of column or columns to be used in regression analysis.
    :type x_column: str, or List[str]
    :param y_column: Name of column to be used as y variable in regression.
    :type y_column: str
    :example:
        >>> from document_filter import DocumentFilter
        >>> from regression import Regression
        >>> doc = DocumentFilter(original_df=cod.our_df, map_choice='mp_e', mode_choice='quad')
        >>> model = Regression(doc_filter=doc, x_column='kills', y_column='placementPercent')
    :note: This will return a Regression object with regression result information.

.. autosummary::
    regression.Regression.r2
    regression.Regression.constant_coefficient
    regression.Regression.x_coefficient
    regression.Regression.lower_confidence
    regression.Regression.upper_confidence
    regression.Regression.pvalue
    regression.Regression.residuals
    regression.Regression.mse
    regression.Regression.ssr
    regression.Regression.ess
    regression.Regression.confidence
    regression.Regression.coefficients

.. _Squad:

Squad
-----
Squad class objects.

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
    :note: *None*

.. autosummary::
    squad.Performance.map
    squad.Performance.mode
    squad.Performance.stats


.. class:: Person:

    The Person class is used to gather all map/mode stats for a given player

    :param original_df: Input data.
    :type original_df: pd.DataFrame
    :param uno: Input person uno Id.
    :type uno: str
    :param gamertag: Input person's gamertag.
    :type gamertag: str
    :example: *None*
    :note: *None*

.. autosummary::
    squad.Person.gamertag
    squad.Person.uno
    squad.Person.rebirth
    squad.Person.verdansk

.. class:: Squad:

    Calculate stats for all maps/modes for each squad memeber.

    :param squad_lst: List of gamertags. Include your gamertag in the list.
    :type squad_lst: List[str]
    :param original_df: Original DataFrame for stats to be calculated from.
    :type original_df: pd.DataFrame
    :param uno_name_dic: A dict of all gamertags and respective unos.
    :type uno_name_dic: dict
    :example:
        >>> from warzone.credentials import user_inputs
            >>> from user import User
            >>> from squad import Squad
            >>> _User = User(info=user_inputs)
            >>> _Squad = Squad(squad_lst=_User.squad_lst, original_df=cod.our_df, uno_name_dic=cod.name_uno_dict)
                >>> from credentials import user_inputs
        >>> from user import User
        >>> from squad import Squad
        >>> _User = User(info=user_inputs)
        >>> _Squad = Squad(squad_lst=_User.squad_lst, original_df=cod.our_df, uno_name_dic=cod.name_uno_dict)
    :note: This will calculate and return the stats for all squad members.

.. autosummary::
    squad.Squad.squad_dic
    squad.Squad.squad_df

.. _User:

User
----
User class objects.

.. :currentmodule:: user

.. class:: User:

    Organizes the Users input data.

    :param info: User input dict.
    :type info: dict
    :example:
        >>> from user import User
        >>> inputs = {'repo': 'local data directory',
        >>>        'gamertag': 'your gamertag',
        >>>        'squad': ['friend gamertag1', 'friend gamertag2', '... etc'],
        >>>        'file_name': 'match_data.csv'}
        >>> user = User(info=inputs)
    :note: *None*


.. autosummary::
    user.User.file_name
    user.User.repo
    user.User.gamertag
    user.User.squad_lst
