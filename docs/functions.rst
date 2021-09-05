.. _Functions:

Functions
*********
.. meta::
   :description: This chapter describes various functions for Analyzing and Visualizing stats.
   :keywords: Call of Duty, Warzone, Python, Data Science

This chapter documents the Functions used in this package.

.. _Analysis:

Analysis
--------
One off functions for various analysis.

.. :currentmodule:: analysis

.. function:: first_top5_bottom_stats(doc_filter, col_lst):

    Calculate mu, std, var, max, min, skew, kurt for all matches depending on **teamPlacement**.
    The intent is for a **map_choice** and **mode_choice** to be fed into the DocumentFilter.
    Does calculations for all matches, regardless of **matchID**.

    :param doc_filter: Input DocumentFilter.
    :type doc_filter: DocumentFilter
    :param col_lst: Input List of Columns to analyze.
    :type col_lst: List[str] or str
    :return: Stats, related to the items in col_lst, for winners, top 5 or 10, and bottom.
    :rtype: pd.DataFrame
    :example: *None*
    :note: If Rebirth is selected in the DocumentFilter, will return top 5. If Verdansk, top 10 is returned.

.. function:: bucket_stats(doc_filter, placement, col_lst):

    Calculate mu, std, var, max, min, skew, kurt for all matches depending on **teamPlacement**.
    The intent is for a **map_choice** and **mode_choice** to be fed into the DocumentFilter.
    Does calculations for all matches, considering of **matchID**.

    :param doc_filter: Input DocumentFilter.
    :type doc_filter: DocumentFilter
    :param placement: Target placement.
    :type placement: List[int] or int
    :param col_lst: Input List of Columns to analyze.
    :type col_lst: List[str] or str
    :return: Stats, related to the items in col_lst, for placement value.
    :rtype: pd.DataFrame
    :example: *None*
    :note: **teamPlacement** value used to filter data. If two int's are provided, will filter within that range.
        First value should be the lower value. Example [0,6] will return top 5 placements.

.. function:: previous_next_placement(doc_filter):

    Calculate mu **teamPlacement** before and after a **teamPlacement**.
    The intent is for a **map_choice** and **mode_choice** to be fed into the DocumentFilter.

    :param doc_filter: Input DocumentFilter.
    :type doc_filter: DocumentFilter
    :return: Previous and next expected placement based on current placement.
    :rtype: pd.DataFrame
    :example: *None*
    :note: *None*

.. function:: match_difficulty(our_doc_filter, other_doc_filter, mu_lst, sum_lst, test):

    Calculate the relative match difficulty based on player and player squad stats.

    :param our_doc_filter: A DocumentFilter with squad and player data only.
    :type our_doc_filter: DocumentFilter
    :param other_doc_filter: A DocumentFilter with all other players data.
    :type other_doc_filter: DocumentFilter
    :param mu_lst: A list of columns to consider the mu. *Optional*
    :type mu_lst: List[str]
    :param sum_lst: A list of columns to consider the sum. *Optional*
    :type sum_lst: List[str]
    :param test: If True, will use all columns for the analysis. *Optional*
    :type test: bool
    :return: Match difficulty.
    :rtype: pd.DataFrame
    :example: *None*
    :note: The intent is for a **map_choice** and **mode_choice** to be fed into both DocumentFilter's.

.. function:: get_daily_hourly_weekday_stats(doc_filter):

    Calculate kills, deaths, wins, top 5s or 10s, match count, and average placement for every day, week, hour.

    :param doc_filter: Input DocumentFilter.
    :type doc_filter: DocumentFilter
    :return: 3 pd.DataFrames and a dict
    :rtype:  *None*
    :example: *None*
    :note: The intent is for a **map_choice** and **mode_choice** to be fed into the DocumentFilter.

.. function:: get_weapons(doc_filter):

    Calculate the Kills, deaths, assists, headshots, average placement and count for each weapon.

    :param doc_filter: Input DocumentFilter.
    :type doc_filter: DocumentFilter
    :return: A DataFrame with a players gun stats.
    :rtype: pd.DataFrame
    :example: *None*
    :note: The intent is for a **username** to be fed into the DocumentFilter and this will return the information for
        that specific player.

.. function:: find_hackers(doc_filter, y_column, col_lst, std):

    Calculate hackers based on various Outlier detection methods.

    :param doc_filter: A DocumentFilter.
    :type doc_filter: DocumentFilter
    :param y_column: A column to consider for Outlier analysis.
    :type y_column: str
    :param col_lst: A list of columns used for Outlier analysis.
    :type col_lst: List[str]
    :param std: The std to be considered for as a threshold, default is 3. *Optional*
    :type std: int
    :return: Returns an index of suspected hackers.
    :rtype: List[int]
    :example: *None*
    :note: The intent is for a **map_choice** and **mode_choice** to be fed into the DocumentFilter.

.. function:: meta_weapons(doc_filter, top_5_or_10, top_1, col, mu):

    Calculate the most popular weapons. Map_choice is required in DocumentFilter if top_5_or_10 or top_1 is True.
    If Neither top_5_or_10 or top_1 are True, it will calculate based on all team placements.
    This will only include loadouts where all attachment slots are filled. This calculates based on a daily interval.

    :param doc_filter: A DocumentFilter.
    :type doc_filter: DocumentFilter
    :param top_5_or_10: If True, will calculate using only the top 5 or 10 place teams, default is False. *Optional*
    :type top_5_or_10: bool
    :param top_1: If True, will calculate using only the 1st place or winning team, default is False. *Optional*
    :type top_1: bool
    :param col: If given will use a column as reference, default is None. None will count gun users per day. *Optional*
    :type col: str
    :param mu: If True, will calculate using mean, default is sum. *Optional*
    :type mu: bool
    :return: The First DataFrame is filled with dict's {kills: 0, deaths: 0, count: 0}.
        The Second is the percent of the lobby using.
    :rtype: List[pd.DataFrame]
    :example: *None*
    :note: *None*

.. _Base:

Base
----
General transformations.

.. :currentmodule:: base

.. function:: normalize(arr, multi):

    Normalize an Array.

    :param arr: Input array.
    :type arr: np.ndarray
    :param multi: If array has multiple columns, default is None. *Optional*
    :type multi: bool
    :return: Normalized array.
    :rtype: np.ndarray
    :example: *None*
    :note: Set *multi* to True, if multiple columns.

.. function:: running_mean(arr, num):

    Calculate the running mean on **num** interval

    :param arr: Input array.
    :type arr: np.ndarray
    :param num: Input int, default is 50. *Optional*
    :type num: int
    :return: Running mean for a given array.
    :rtype: np.ndarray
    :example: *None*
    :note: *None*

.. function:: cumulative_mean(arr):

    Calculate the cumulative mean.

    :param arr: Input array.
    :type arr: np.ndarray
    :return: Cumulative mean for a given array.
    :rtype: np.ndarray
    :example: *None*
    :note: *None*

.. _Build:

Build
-----
These functions are used when building the CallofDuty class.

:ref:`CallofDuty <CallofDuty>`

.. _Outlier:

Outlier
-------
Various outlier detection functions.

.. :currentmodule:: outlier

.. function:: stack(x_arr, y_arr, multi):

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

.. function:: _cent(x_lst, y_lst):

    Calculate the centroid from x and y value(s).

    :param x_lst: A list of values.
    :type x_lst: List[float]
    :param y_lst: A list of values.
    :type y_lst: List[float]
    :returns: A list of x and y values representing the centroid of two lists.
    :rtype: List[float]
    :example: *None*
    :note: *None*

.. function:: _dis(cent1, cent2):

    Calculate distance between two centroids.

    :param cent1: An x, y coordinate representing a centroid.
    :type cent1: List[float]
    :param cent2: An x, y coordinate representing a centroid.
    :type y_lst: List[float]
    :returns: A distance measurement.
    :rtype: float
    :example: *None*
    :note: *None*

.. function:: outlier_std(arr, data, y_column, _std, plus):

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

.. function:: outlier_var(arr, data, y_column, per, plus):

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

.. function:: outlier_regression(arr, data, x_column, y_column, _std, plus):

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

.. function:: outlier_distance(arr, data, x_column, y_column, _std, plus):

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

.. function:: outlier_hist(arr, data, x_column, per, plus):

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

.. function:: outlier_knn(arr, data, x_column, y_column, _std, plus):

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

.. function:: outlier_cooks_distance(arr, data, x_column, y_column, plus, return_df):

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

.. _Plots:

Plots
------
Various one off plots.

.. :currentmodule:: plots

.. function:: personal_plot(doc_filter):

    Returns a series of plots.

    :param doc_filter: A DocumentFilter.
    :type doc_filter: DocumentFilter
    :return: *None*
    :example: *None*
    :note: This is intended to be used with **map_choice**, **mode_choice** and a **username** inputted into the DocumentFilter.

.. function:: lobby_plot(doc_filter):

    Returns a series of plots.

    :param doc_filter: A DocumentFilter.
    :type doc_filter: DocumentFilter
    :return: *None*
    :example: *None*
    :note: This is intended to be used with **map_choice** and **mode_choice** inputted into the DocumentFilter.

.. function:: squad_plot(doc_filter, col_lst):

    Build a Polar plot for visualizing squad stats.

    :param doc_filter: A DocumentFilter.
    :type doc_filter: DocumentFilter
    :param col_lst: Input List of Columns to analyze.
    :type col_lst: List[str] or str
    :return: *None*
    :example: *None*
    :note: This is intended to be used with **map_choice** and **mode_choice** inputted into the DocumentFilter.

.. _Scrape:

Scrape
------
Functions for getting and dealing with new data.

`Getting Data <https://medium.com/@peterjrigali/warzone-package-part-1-b64d753e949c>`_

.. :currentmodule:: scrape

.. function:: connect_to_api(_id):

    Connect to Call of Duty API.

    :param _id: A matchID str.
    :type _id: str
    :return: A Json of lobby data related to specified matchID.
    :rtype: Json
    :example: *None*
    :note: Connect to Cod API to receive lobby information.

.. function:: clean_api_data(json_object):

    Cleans the JSON output from **connect_to_api**

    :param json_object: Json object.
    :type json_object: Json
    :return: Match information in a table.
    :rtype: pd.DataFrame
    :example: *None*
    :note: Takes a Json object related to a **matchID** and constructs a pd.DataFrame with all relevant information.
        This will need to be saved(or concatenated to an existing csv) and
        loaded through the _evaluate_df() to work properly in this model.
