import pandas as pd
from dataclasses import dataclass
from warzone.document_filter import DocumentFilter
from warzone.gun_dictionary import gun_dict

MU_LST = ['headshots', 'kills', 'deaths', 'longestStreak', 'scorePerMinute', 'distanceTraveled',
          'percentTimeMoving', 'damageDone', 'damageTaken', 'missionsComplete', 'timePlayed', 'objectiveBrCacheOpen',
          'objectiveBrKioskBuy', 'objectiveBrMissionPickupTablet', 'objectiveLastStandKill', 'objectiveReviver',
          'objectiveTeamWiped', 'objectiveLastStandKill', 'objectiveBrDownEnemyCircle1',
          'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3', 'objectiveBrDownEnemyCircle4',
          'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6', 'placementPercent', 'headshotRatio']
"""A list of columns to compute the mean for"""

SUM_LST = ['headshots', 'kills', 'deaths', 'distanceTraveled', 'damageDone', 'damageTaken', 'missionsComplete',
           'timePlayed', 'objectiveBrCacheOpen', 'objectiveBrKioskBuy', 'objectiveBrMissionPickupTablet',
           'objectiveLastStandKill', 'objectiveReviver', 'objectiveTeamWiped', 'objectiveLastStandKill',
           'objectiveBrDownEnemyCircle1', 'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3',
           'objectiveBrDownEnemyCircle4', 'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6']
"""A list of columns to compute the sum for"""

MAX_LST = ['headshots', 'kills', 'deaths', 'distanceTraveled', 'damageDone', 'damageTaken', 'missionsComplete',
           'timePlayed', 'objectiveBrCacheOpen', 'objectiveBrKioskBuy', 'objectiveBrMissionPickupTablet',
           'objectiveLastStandKill', 'objectiveReviver', 'objectiveTeamWiped', 'objectiveLastStandKill',
           'objectiveBrDownEnemyCircle1', 'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3',
           'objectiveBrDownEnemyCircle4', 'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6', 'kdRatio',
           'headshotRatio']
"""A list of columns to compute the max for"""


def get_indexes(original_df: pd.DataFrame, uno: str):
    lst = original_df["uno"].tolist()
    return tuple([ind for ind, val in enumerate(lst) if val == uno])


def _build_stats(df: pd.DataFrame, stat_type: str):
    if stat_type == "sum":
        return {col: df[col].sum() for col in SUM_LST}
    elif stat_type == "mean":
        return {col: df[col].mean() for col in MU_LST}
    elif stat_type == "median":
        return {col: df[col].median() for col in MU_LST}
    elif stat_type == "max":
        return {col: df[col].max() for col in MAX_LST}


def _build_weapons(df: pd.DataFrame, stat_type: str):
    loadouts = df["loadouts"]
    excluded_weapons = {"iw8_fists": True, "none": True, "nan": True}
    gun_ind_dic = {key: [] for key in gun_dict.keys()}
    for ind, row in enumerate(loadouts):
        for loadout in row:
            for weapon in loadout:
                if weapon not in excluded_weapons:
                    gun_ind_dic[weapon].append(ind)

    dic = {key: {'kills': 0, 'deaths': 0, 'headshots': 0, 'assists': 0} for key in gun_ind_dic.keys()}
    for key, val in gun_ind_dic.items():
        temp_data = df.iloc[val]
        if temp_data.empty:
            continue
        else:
            if stat_type == "sum":
                for col in ['kills', 'deaths', 'headshots', 'assists']:
                    dic[key][col] = temp_data[col].sum()
            elif stat_type == "mean":
                for col in ['kills', 'deaths', 'headshots', 'assists']:
                    dic[key][col] = temp_data[col].mean()
            elif stat_type == "median":
                for col in ['kills', 'deaths', 'headshots', 'assists']:
                    dic[key][col] = temp_data[col].median()
            elif stat_type == "max":
                for col in ['kills', 'deaths', 'headshots', 'assists']:
                    dic[key][col] = temp_data[col].max()

            if dic[key]['kills'] != 0 and dic[key]['deaths'] != 0:
                dic[key]['kdRatio'] = dic[key]['kills'] / dic[key]['deaths']
            else:
                dic[key]['kdRatio'] = 0.0
            if dic[key]['kills'] != 0 and dic[key]['headshots'] != 0:
                dic[key]['headshotRatio'] = dic[key]['headshots'] / dic[key][
                    'kills']
            else:
                dic[key]['headshotRatio'] = 0.0
            dic[key]['averagePlacementPercent'] = temp_data['placementPercent'].mean()
            dic[key]['count'] = len(temp_data)
            if '_' in key:
                dic[key]['weaponType'] = key.split('_')[1]
            else:
                dic[key]['weaponType'] = 'None'
    return {gun_dict[key]: dic[key] for key in dic.keys()}


def _build_time(df: pd.DataFrame, stat_type: str):
    seconds = df['timePlayed']
    minutes = seconds / 60
    hours = minutes / 60
    days = hours / 24

    if stat_type == "sum":
        return {"seconds": seconds.sum(), "minutes": minutes.sum(), "hours": hours.sum(), "days": days.sum()}
    elif stat_type == "mean":
        return {"seconds": seconds.mean(), "minutes": minutes.mean(), "hours": hours.mean(), "days": days.mean()}
    elif stat_type == "median":
        return {"seconds": seconds.median(), "minutes": minutes.median(), "hours": hours.median(), "days": days.median()}
    elif stat_type == "max":
        return {"seconds": seconds.max(), "minutes": minutes.max(), "hours": hours.max(), "days": days.max()}


@dataclass
class Person:
    """

    The Person class is used to gather all map/mode stats for a given player

    :param original_df: Input data.
    :type original_df: pd.DataFrame
    :param uno: Input person uno Id.
    :type uno: str
    :param gamertag: Input person's gamertag.
    :type gamertag: str
    :example: *None*
    :note: *None

    """

    __slots__ = ["uno", "gamertag", "index_lst", "performance_dict", "favorite_dict", "_original_df"]

    def __init__(self, original_df: pd.DataFrame, uno: str, gamertag: str):
        self.uno = uno
        self.gamertag = gamertag
        self.index_lst = get_indexes(original_df=original_df, uno=uno)
        self.performance_dict = None
        self.favorite_dict = None
        self._original_df = original_df

    def get_performance(self, map_choice, mode_choice, team_size, stat_type: str = 'sum',
                        stats_weapon_time: str = "stats", position: str = "all", favorite: bool = False) -> dict:
        df = DocumentFilter(input_df=self._original_df.iloc[list(self.index_lst)], map_choice=map_choice,
                            mode_choice=mode_choice, team_size=team_size, position=position).df
        if stats_weapon_time == "stats":
            dic = _build_stats(df=df, stat_type=stat_type)
        elif stats_weapon_time == "weapons":
            dic = _build_weapons(df=df, stat_type=stat_type)
        elif stats_weapon_time == "time":
            dic = _build_time(df=df, stat_type=stat_type)
        else:
            raise AttributeError("stat_type must be one of the following strings (stats, weapons, time)")

        if favorite:
            self.favorite_dict[map_choice][mode_choice][team_size][stats_weapon_time][stat_type] = dic
        else:
            self.performance_dict[map_choice][mode_choice][team_size][stats_weapon_time][stat_type] = dic
        return dic

    def __repr__(self):
        return self.gamertag
