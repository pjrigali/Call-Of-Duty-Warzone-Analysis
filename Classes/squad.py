from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from Classes.document_filter import DocumentFilter

mu_lst = ['headshots', 'kills', 'deaths', 'kdRatio', 'longestStreak', 'scorePerMinute', 'distanceTraveled',
          'percentTimeMoving', 'damageDone', 'damageTaken', 'missionsComplete', 'timePlayed', 'objectiveBrCacheOpen',
          'objectiveBrKioskBuy', 'objectiveBrMissionPickupTablet', 'objectiveLastStandKill', 'objectiveReviver',
          'objectiveTeamWiped', 'objectiveLastStandKill', 'objectiveBrDownEnemyCircle1',
          'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3', 'objectiveBrDownEnemyCircle4',
          'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6', 'placementPercent', 'headshotRatio']

sum_lst = ['headshots', 'kills', 'deaths', 'distanceTraveled', 'damageDone', 'damageTaken', 'missionsComplete',
           'timePlayed', 'objectiveBrCacheOpen', 'objectiveBrKioskBuy', 'objectiveBrMissionPickupTablet',
           'objectiveLastStandKill', 'objectiveReviver', 'objectiveTeamWiped', 'objectiveLastStandKill',
           'objectiveBrDownEnemyCircle1', 'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3',
           'objectiveBrDownEnemyCircle4', 'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6']

max_lst = ['headshots', 'kills', 'deaths', 'distanceTraveled', 'damageDone', 'damageTaken', 'missionsComplete',
           'timePlayed', 'objectiveBrCacheOpen', 'objectiveBrKioskBuy', 'objectiveBrMissionPickupTablet',
           'objectiveLastStandKill', 'objectiveReviver', 'objectiveTeamWiped', 'objectiveLastStandKill',
           'objectiveBrDownEnemyCircle1', 'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3',
           'objectiveBrDownEnemyCircle4', 'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6', 'kdRatio',
           'headshotRatio']

lst_dic = {'mu': mu_lst, 'sum': sum_lst, 'max': max_lst}


def _get_stats(doc_filter: DocumentFilter) -> Dict[str, float]:
    df = doc_filter.df.fillna(0)
    stats_dic = {'game_count': len(df), 'win_count': len(df[df['teamPlacement'] == 1]), 'win_percent': 0.0}
    if stats_dic['win_count'] != 0:
        stats_dic['win_percent'] = stats_dic['win_count'] / stats_dic['game_count']

    if doc_filter.map_choice == 'mp_d':
        stats_dic['top_10_count'] = len(df[(df['teamPlacement'] <= 10) & (df['teamPlacement'] > 1)])
        stats_dic['top_10_percent'] = 0.0
        if stats_dic['top_10_count'] != 0:
            stats_dic['top_10_percent'] = stats_dic['top_10_count'] / stats_dic['game_count']
    else:
        stats_dic['top_5_count'] = len(df[(df['teamPlacement'] <= 10) & (df['teamPlacement'] > 1)])
        stats_dic['top_5_percent'] = 0.0
        if stats_dic['top_5_count'] != 0:
            stats_dic['top_5_percent'] = stats_dic['top_5_count'] / stats_dic['game_count']

    for lst in lst_dic.keys():
        col_lst = lst_dic[lst]
        if lst == 'mu':
            for col in col_lst:
                stats_dic['average_' + col] = np.mean(df[col])
            continue
        elif lst == 'sum':
            for col in col_lst:
                stats_dic['total_' + col] = np.sum(df[col])
            continue
        elif lst == 'max':
            for col in col_lst:
                stats_dic['max_' + col] = np.max(df[col])

    return {str(i): float(stats_dic[i]) for i in stats_dic.keys()}


class Performance:

    original_df: pd.DataFrame
    map_choice: str
    mode_choice: str
    uno: str

    def __init__(self, original_df, map_choice, mode_choice, uno):
        self._map = map_choice
        self._mode = mode_choice
        self._stats = _get_stats(doc_filter=DocumentFilter(original_df=original_df,
                                                           map_choice=map_choice,
                                                           mode_choice=mode_choice,
                                                           uno=uno))

    def __repr__(self):
        return self._mode

    @property
    def map(self) -> str:
        return self._map

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def stats(self) -> dict:
        return self._stats


def _get_stats_per_map(map_choice: str, original_df: pd.DataFrame, uno: str) -> Dict[str, Performance]:
    mode_lst = ['solo', 'duo', 'trio', 'quad']
    return {_mode: Performance(original_df=original_df,
                               map_choice=map_choice,
                               mode_choice=_mode,
                               uno=uno) for _mode in mode_lst}


class Person:

    original_df: pd.DataFrame
    uno: str
    gamertag: str

    def __init__(self, original_df, uno, gamertag):
        self._uno = uno
        self._gamertag = gamertag
        self._rebirth_stats = _get_stats_per_map(map_choice='mp_e', original_df=original_df, uno=uno)
        self._verdansk_stats = _get_stats_per_map(map_choice='mp_d', original_df=original_df, uno=uno)

    def __repr__(self):
        return self.gamertag

    @property
    def gamertag(self):
        return self._gamertag

    @property
    def uno(self):
        return self._uno

    @property
    def rebirth(self):
        return self._rebirth_stats

    @property
    def verdansk(self):
        return self._verdansk_stats


class Squad:

    squad_lst: List[str]
    original_df: pd.DataFrame
    uno_name_dic: Dict[str, str]

    def __init__(self, squad_lst, original_df, uno_name_dic):
        self._squad_stats = {teammate: Person(gamertag=teammate,
                                              original_df=original_df,
                                              uno=uno_name_dic[teammate]) for teammate in squad_lst}
        lst = []
        for teammate in squad_lst:
            for _mode in ['solo', 'duo', 'trio', 'quad']:
                self._squad_stats[teammate].rebirth[_mode].stats['map'] = 'mp_e'
                self._squad_stats[teammate].verdansk[_mode].stats['map'] = 'mp_d'
                self._squad_stats[teammate].rebirth[_mode].stats['mode'] = _mode
                self._squad_stats[teammate].verdansk[_mode].stats['mode'] = _mode
                self._squad_stats[teammate].rebirth[_mode].stats['username'] = teammate
                self._squad_stats[teammate].verdansk[_mode].stats['username'] = teammate
                lst.append(self._squad_stats[teammate].rebirth[_mode].stats)
                lst.append(self._squad_stats[teammate].verdansk[_mode].stats)
        self._squad_df = pd.DataFrame(lst).fillna(0.0)

    def __repr__(self):
        return 'Squad Stats'

    @property
    def squad_dic(self) -> Dict[str, Person]:
        return self._squad_stats

    @property
    def squad_df(self) -> pd.DataFrame:
        return self._squad_df
