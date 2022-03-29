"""User class object.

Usage:
 ./warzone/classes/time_series.py

Author:
 Peter Rigali - 2021-08-30
"""
from dataclasses import dataclass
from typing import List, Optional, Dict
import datetime
import pandas as pd
from warzone.classes.document_filter import DocumentFilter
from warzone.analysis import match_difficulty

_mu_lst = ['headshots', 'kills', 'deaths', 'longestStreak', 'scorePerMinute', 'distanceTraveled',
           'percentTimeMoving', 'damageDone', 'damageTaken', 'missionsComplete', 'timePlayed',
           'objectiveBrCacheOpen',
           'objectiveBrKioskBuy', 'objectiveBrMissionPickupTablet', 'objectiveLastStandKill', 'objectiveReviver',
           'objectiveTeamWiped', 'objectiveLastStandKill', 'objectiveBrDownEnemyCircle1',
           'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3', 'objectiveBrDownEnemyCircle4',
           'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6', 'placementPercent', 'headshotRatio']
_sum_lst = ['headshots', 'kills', 'deaths', 'distanceTraveled', 'damageDone', 'damageTaken', 'missionsComplete',
            'timePlayed', 'objectiveBrCacheOpen', 'objectiveBrKioskBuy', 'objectiveBrMissionPickupTablet',
            'objectiveLastStandKill', 'objectiveReviver', 'objectiveTeamWiped', 'objectiveLastStandKill',
            'objectiveBrDownEnemyCircle1', 'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3',
            'objectiveBrDownEnemyCircle4', 'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6']


def get_sessions(data: pd.DataFrame, minutes: int = 60) -> Dict[int, pd.DataFrame]:
    """Splits games into sessions based on a threshold between games"""
    id_dic, ind_dic, count, past_game = {}, {}, 0, None
    for ind, row in data.iterrows():
        if past_game is None:
            past_game, id_dic[row['matchID']], ind_dic[count] = row['endDateTime'], True, [ind]
            continue
        elif row['matchID'] in id_dic:
            ind_dic[count].append(ind)
        else:
            if row['startDateTime'] - past_game <= datetime.timedelta(minutes=minutes):
                past_game, id_dic[row['matchID']] = row['endDateTime'], True
                ind_dic[count].append(ind)
            else:
                count += 1
                past_game, ind_dic[count] = row['endDateTime'], [ind]
    session_df_dic = {key: data.iloc[val] for key, val in ind_dic.items()}
    return session_df_dic


def get_group_stats(args) -> Dict[str, pd.DataFrame]:
    """Returns a dict with calculations for a given session"""
    data, lower, upper, mu_lst, sum_lst, quantile_lst, median_lst = args
    group = data.groupby('startDateTime')
    t = group.last()[['difficulty', 'ourPlacement']]
    return {'mu': pd.concat([group.mean()[_mu_lst].sort_index(ascending=True), t], axis=1),
            'sum': pd.concat([group.sum()[_sum_lst].sort_index(ascending=True), t], axis=1),
            'median': pd.concat([group.median()[median_lst].sort_index(ascending=True), t], axis=1),
            'lower': pd.concat([group.quantile(q=lower)[quantile_lst].sort_index(ascending=True), t], axis=1),
            'upper': pd.concat([group.quantile(q=upper)[quantile_lst].sort_index(ascending=True), t], axis=1),
            'raw': data}


@dataclass
class TSanalysis:
    """

    Builds TSanalysis Class. Used for time series analysis of matches.

    :param our_doc_filter: Input DocumentFilter of players data.
    :type our_doc_filter: DocumentFilter
    :param other_doc_filter: Input DocumentFilter of other players data.
    :type other_doc_filter: DocumentFilter
    :param lower: Lower quantile value. Default is 0.159 *Optional*
    :type lower: float
    :param upper: Upper quantile value. Default is 0.841 *Optional*
    :type upper: float
    :param use_sessions: If True, will segment the data into sessions. *Optional*
    :type use_sessions: bool
    :param mu_lst: List of columns to find the Mean. There is a deafult Mu list built in. *Optional*
    :type mu_lst: list
    :param sum_lst: List of columns to find the Sum. There is adefault Sum list built in *Optional*
    :type sum_lst: list
    :param quantile_lst: List of columns to find the Quantile. *Optional*
    :type quantile_lst: list
    :param median_lst: List of columns to find the Median. Default is mu_lst. *Optional*
    :type median_lst: list
    :param session_threshold_time: Amount of lag allowable between gaming sessions. In minutes, default is 60 *Optional*
    :type session_threshold_time: int
    :example: *None*
    :note: *None*

    """
    def __init__(self, our_doc_filter: DocumentFilter, other_doc_filter: DocumentFilter, lower: Optional[float] = 0.159,
                 upper: Optional[float] = 0.841, use_sessions: bool = True, mu_lst: Optional[list] = None,
                 sum_lst: Optional[list] = None, quantile_lst: Optional[list] = None,
                 median_lst: Optional[list] = None, session_threshold_time: int = 60):
        self._lower = lower
        self._upper = upper
        self._minutes = session_threshold_time

        self._mu_lst = mu_lst
        if mu_lst is None:
            self._mu_lst = _mu_lst

        self._sum_lst = sum_lst
        if sum_lst is None:
            self._sum_lst = _sum_lst

        self._quantile_lst = quantile_lst
        if quantile_lst is None:
            self._quantile_lst = _mu_lst

        self._median_lst = median_lst
        if median_lst is None:
            self._median_lst = _mu_lst

        self._match_difficulty = match_difficulty(our_doc_filter=our_doc_filter, other_doc_filter=other_doc_filter)
        self._len = self._match_difficulty.shape[0]
        self._startDateTimes = our_doc_filter.df['startDateTime'].unique().tolist()

        difficulty_dic = self._match_difficulty.to_dict()['difficulty']
        placement_dic = self._match_difficulty.to_dict()['ourPlacement']
        our_df = our_doc_filter.df.copy()
        other_df = other_doc_filter.df.copy()
        for i in [our_df, other_df]:
            lst = i['matchID'].tolist()
            i['difficulty'] = [difficulty_dic[id] for id in lst]
            i['ourPlacement'] = [placement_dic[id] for id in lst]

        self._our_dic = {}
        self._other_dic = {}
        lst = (self._lower, self._upper, self._mu_lst, self._sum_lst, self._quantile_lst, self._median_lst)
        if use_sessions:
            for dic, df in [(self._our_dic, our_df), (self._other_dic, other_df)]:
                temp = get_sessions(data=df, minutes=session_threshold_time)
                for key, val in temp.items():
                    dic[key] = get_group_stats(args=(val,) + lst)
        else:
            self._our_dic = {0: get_group_stats(args=(our_df,) + lst)}
            self._other_dic = {0: get_group_stats(args=(other_df,) + lst)}

    def __getitem__(self, key):
        return self._our_dic[key], self._other_dic[key]

    def __len__(self):
        return self._len

    def __repr__(self):
        return 'TSanalysis'

    @property
    def mu_lst(self) -> List[str]:
        """Returns list of items to calculate the mean"""
        return self._mu_lst

    @property
    def sum_lst(self) -> List[str]:
        """Returns list of items to calculate the sum"""
        return self._sum_lst

    @property
    def quantile_lst(self) -> List[str]:
        """Returns list of items to calculate the quantile"""
        return self._quantile_lst

    @property
    def median_lst(self) -> List[str]:
        """Returns list of items to calculate the median"""
        return self._median_lst

    @property
    def match_difficulty(self) -> pd.DataFrame:
        """Returns DataFrame of match difficulty"""
        return self._match_difficulty

    @property
    def start_times(self) -> list:
        """Returns list of match start times"""
        return self._startDateTimes

    @property
    def our_data(self) -> dict:
        """Returns dict of our data"""
        return self._our_dic

    @property
    def other_data(self) -> dict:
        """Returns dict of other data"""
        return self._other_dic

# our = ts_data.our_data[0]['median']['kills'].tolist()
# other = ts_data.other_data[0]['median']['kills'].tolist()

# Euclidian Distance
# t = math.dist(our, other) / len(our)

# MAPE
# can not handle zero in denominator
# def mape(actual, pred):
#     actual = np.array([i if i != 0 else .01 for i in actual])
#     pred = np.array([i if i != 0 else .01 for i in pred])
#     return np.mean(np.abs((actual - pred) / actual)) * 100
# t = mape(our, other)

# Auto-Correlation
# def acf(x, length=50):
#     return [1]+[np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, length)]
# t = acf(our)
# tt = acf(other)
# ttt = np.corrcoef(t, tt)[0, 1]

# Correlation
# t = np.corrcoef(our, other)[0, 1]

# Dynamic Time Warping
# from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean
#
# our_stack = np.zeros((len(our), 2))
# our_stack[:, 0] = our
# other_stack = np.zeros((len(our), 2))
# other_stack[:, 0] = other
#
# distance_matrix = fastdtw(our_stack, other_stack, dist=euclidean)[0]

# from pyts.approximation import SymbolicAggregateApproximation
# from pyts.transformation import BagOfPatterns
# sax = SymbolicAggregateApproximation(n_bins=6, strategy='normal')
# X_sax_our = sax.fit_transform(our_stack)
# X_sax_other = sax.fit_transform(other_stack)
# t = ''.join(X_sax_our[:, 0])
# tt = ''.join(X_sax_other[:, 0])
# X = np.array([our, other])
# bop = BagOfPatterns(window_size=50, word_size=2, n_bins=6,sparse=False)
# X_bop = bop.fit_transform(X)
# t = bop.vocabulary_