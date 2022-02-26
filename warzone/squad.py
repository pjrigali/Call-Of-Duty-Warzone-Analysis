"""Squad class object.

Usage:
 ./squad.py

Author:
 Peter Rigali - 2021-08-30
"""
from typing import List, Dict
import pandas as pd
import numpy as np
from dataclasses import dataclass
from warzone.document_filter import DocumentFilter
from warzone.gun_dictionary import gun_dict

mu_lst = ['headshots', 'kills', 'deaths', 'longestStreak', 'scorePerMinute', 'distanceTraveled',
          'percentTimeMoving', 'damageDone', 'damageTaken', 'missionsComplete', 'timePlayed', 'objectiveBrCacheOpen',
          'objectiveBrKioskBuy', 'objectiveBrMissionPickupTablet', 'objectiveLastStandKill', 'objectiveReviver',
          'objectiveTeamWiped', 'objectiveLastStandKill', 'objectiveBrDownEnemyCircle1',
          'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3', 'objectiveBrDownEnemyCircle4',
          'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6', 'placementPercent', 'headshotRatio']
"""A list of columns to compute the mean for"""

sum_lst = ['headshots', 'kills', 'deaths', 'distanceTraveled', 'damageDone', 'damageTaken', 'missionsComplete',
           'timePlayed', 'objectiveBrCacheOpen', 'objectiveBrKioskBuy', 'objectiveBrMissionPickupTablet',
           'objectiveLastStandKill', 'objectiveReviver', 'objectiveTeamWiped', 'objectiveLastStandKill',
           'objectiveBrDownEnemyCircle1', 'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3',
           'objectiveBrDownEnemyCircle4', 'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6']
"""A list of columns to compute the sum for"""

max_lst = ['headshots', 'kills', 'deaths', 'distanceTraveled', 'damageDone', 'damageTaken', 'missionsComplete',
           'timePlayed', 'objectiveBrCacheOpen', 'objectiveBrKioskBuy', 'objectiveBrMissionPickupTablet',
           'objectiveLastStandKill', 'objectiveReviver', 'objectiveTeamWiped', 'objectiveLastStandKill',
           'objectiveBrDownEnemyCircle1', 'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3',
           'objectiveBrDownEnemyCircle4', 'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6', 'kdRatio',
           'headshotRatio']
"""A list of columns to compute the max for"""

lst_dic = {'mu': mu_lst, 'sum': sum_lst, 'max': max_lst}


def _get_stats(doc_filter: DocumentFilter) -> dict:
    """Calculates the stats for a given DocumentFilter"""

    df = doc_filter.df.fillna(0)
    stats_dic = {'game_count': len(df), 'win_count': len(df[df['teamPlacement'] == 1]), 'win_percent': 0.0}
    if stats_dic['win_count'] != 0:
        stats_dic['win_percent'] = stats_dic['win_count'] / stats_dic['game_count']

    if doc_filter.mode_choice == 'royale':
        stats_dic['top_10_count'] = len(df[(df['teamPlacement'] <= 10) & (df['teamPlacement'] > 1)])
        stats_dic['top_10_percent'] = 0.0
        if stats_dic['top_10_count'] != 0:
            stats_dic['top_10_percent'] = stats_dic['top_10_count'] / stats_dic['game_count']
    elif doc_filter.mode_choice == 'resurgence':
        stats_dic['top_5_count'] = len(df[(df['teamPlacement'] <= 10) & (df['teamPlacement'] > 1)])
        stats_dic['top_5_percent'] = 0.0
        if stats_dic['top_5_count'] != 0:
            stats_dic['top_5_percent'] = stats_dic['top_5_count'] / stats_dic['game_count']

    if sum(df['kills'].tolist()) > 0:
        temp_kd = sum(df['kills'].tolist()) / sum(df['deaths'].tolist())
    else:
        temp_kd = 0.0
    stats_dic['average_kdRatio'] = temp_kd

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

    temp_dic = {str(i): float(stats_dic[i]) for i in stats_dic.keys()}
    if stats_dic['total_timePlayed'] > 0.0:
        temp_dic['total_secondsPlayed'] = stats_dic['total_timePlayed']
        temp_dic['total_minutesPlayed'] = temp_dic['total_secondsPlayed'] / 60
        temp_dic['total_hoursPlayed'] = temp_dic['total_minutesPlayed'] / 60
        temp_dic['total_daysPlayed'] = temp_dic['total_hoursPlayed'] / 24
    else:
        temp_dic['total_secondsPlayed'] = 0.0
        temp_dic['total_minutesPlayed'] = 0.0
        temp_dic['total_hoursPlayed'] = 0.0
        temp_dic['total_daysPlayed'] = 0.0
    return temp_dic


@dataclass
class Performance:
    """

    The Performance class is used to evaluate a players performance on a given map and mode

    :param original_df: Input data.
    :type original_df: pd.DataFrame
    :param map_choice: Map filter. Either 'rebirth', 'verdansk' or 'caldera'.
    :type map_choice: str
    :param mode_choice: Mode filter. Either 'royale' or 'resurgence'.
    :type mode_choice: str
    :param team_size:  Team size filter. Either 'solo', 'duo', 'trio', or 'quad'.
    :type team_size: str
    :param uno: Input person uno Id.
    :type uno: str
    :example: *None*
    :note: *None

    """

    original_df: pd.DataFrame
    map_choice: str
    mode_choice: str
    # team_size: str
    uno: str

    def __init__(self, original_df, map_choice, mode_choice, team_size, uno):
        self._map = map_choice
        self._mode = mode_choice
        self._team_size = team_size
        self._stats = _get_stats(doc_filter=DocumentFilter(input_df=original_df,
                                                           map_choice=map_choice,
                                                           mode_choice=mode_choice,
                                                           team_size=team_size,
                                                           uno=uno))

    def __repr__(self):
        return self._mode

    @property
    def map(self) -> str:
        """Returns the map selected"""
        return self._map

    @property
    def mode(self) -> str:
        """Returns the mode selected"""
        return self._mode

    @property
    def team_size(self) -> str:
        """Returns the team size selected"""
        return self._team_size

    @property
    def stats(self) -> dict:
        """Returns a dict of stats"""
        return self._stats


def _get_stats_per_map(map_choice: str, original_df: pd.DataFrame, uno: str) -> dict:
    """Calculates the stats for all modes on a given map"""
    final_dic = {}
    for _mode in ['royale', 'resurgence']:
        final_dic[_mode] = {}
        for _team_size in ['solo', 'duo', 'trio', 'quad']:
            final_dic[_mode][_team_size] = Performance(original_df=original_df,
                                                       map_choice=map_choice,
                                                       mode_choice=_mode,
                                                       team_size=_team_size,
                                                       uno=uno)
    return final_dic


def _get_time_played(data: dict) -> dict:
    """Calculates time played for each map"""
    final_dic = {}
    for _mode in ['royale', 'resurgence']:
        final_dic[_mode] = {}
        for _team_size in ['solo', 'duo', 'trio', 'quad']:
            final_dic[_mode][_team_size] = {'days': 0.0, 'hours': 0.0, 'minutes': 0.0, 'seconds': 0.0}
            final_dic[_mode][_team_size]['seconds'] += data[_mode][_team_size].stats['total_secondsPlayed']
            final_dic[_mode][_team_size]['minutes'] += data[_mode][_team_size].stats['total_minutesPlayed']
            final_dic[_mode][_team_size]['hours'] += data[_mode][_team_size].stats['total_hoursPlayed']
            final_dic[_mode][_team_size]['days'] += data[_mode][_team_size].stats['total_daysPlayed']
    return final_dic


def _get_weapon_data(original_df: pd.DataFrame, uno: str) -> dict:
    """Calculates weapon stats for resurgence and royale"""
    excluded_weapons = {"iw8_fists": True, "none": True, "nan": True}
    included_wzgunname_dic = {}
    included_gunname_dic = {}
    for wz_gun_name in gun_dict.keys():
        if wz_gun_name not in excluded_weapons:
            included_wzgunname_dic[wz_gun_name] = True
            included_gunname_dic[gun_dict[wz_gun_name]] = True

    gun_data_dic = {}
    for _mode in ['resurgence', 'royale']:
        data = DocumentFilter(input_df=original_df, mode_choice=_mode, uno=uno).df
        gun_ind_dic = {gun: set() for gun in included_gunname_dic.keys()}
        for ind, person in enumerate(data['loadouts'].tolist()):
            for loadout in person:
                primary, secondary = loadout[0], loadout[1]
                if primary != "iw8_fists" and secondary != "iw8_fists":
                    if primary in included_wzgunname_dic:
                        gun_ind_dic[gun_dict[primary]].add(ind)
                    if secondary in included_wzgunname_dic:
                        gun_ind_dic[gun_dict[secondary]].add(ind)
        # Get weapon stats
        weapon_type_dic = dict(zip(included_gunname_dic.keys(), included_wzgunname_dic.keys()))
        gun_stats_dic = {gun: {'kills': 0, 'deaths': 0, 'headshots': 0, 'assists': 0} for gun in
                         list(gun_dict.values())}
        for key, value in gun_ind_dic.items():
            temp_df = data.iloc[list(value)]
            for col in ['kills', 'deaths', 'headshots', 'assists']:
                gun_stats_dic[key][col] = temp_df[col].sum()
            if gun_stats_dic[key]['kills'] != 0 and gun_stats_dic[key]['deaths'] != 0:
                gun_stats_dic[key]['kdRatio'] = gun_stats_dic[key]['kills'] / gun_stats_dic[key]['deaths']
            else:
                gun_stats_dic[key]['kdRatio'] = 0.0
            if gun_stats_dic[key]['kills'] != 0 and gun_stats_dic[key]['headshots'] != 0:
                gun_stats_dic[key]['headshotRatio'] = gun_stats_dic[key]['headshots'] / gun_stats_dic[key]['kills']
            else:
                gun_stats_dic[key]['headshotRatio'] = 0.0
            gun_stats_dic[key]['averagePlacementPercent'] = temp_df['placementPercent'].mean()
            gun_stats_dic[key]['count'] = len(temp_df)
            if '_' in weapon_type_dic[key]:
                gun_stats_dic[key]['weaponType'] = weapon_type_dic[key].split('_')[1]
            else:
                gun_stats_dic[key]['weaponType'] = 'None'
            gun_data_dic[_mode] = pd.DataFrame.from_dict(gun_stats_dic, orient='index').fillna(0.0)
    return gun_data_dic


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
    original_df: pd.DataFrame
    uno: str
    gamertag: str

    def __init__(self, original_df, uno, gamertag):
        self._uno = uno
        self._gamertag = gamertag
        self._rebirth_stats = _get_stats_per_map(map_choice='rebirth', original_df=original_df, uno=uno)
        self._verdansk_stats = _get_stats_per_map(map_choice='verdansk', original_df=original_df, uno=uno)
        self._caldera_stats = _get_stats_per_map(map_choice='caldera', original_df=original_df, uno=uno)
        self._rebirth_time = _get_time_played(data=self._rebirth_stats)
        self._verdansk_time = _get_time_played(data=self._verdansk_stats)
        self._caldera_time = _get_time_played(data=self._caldera_stats)
        self._weapon_data = _get_weapon_data(original_df=original_df, uno=uno)

    def __repr__(self):
        return self.gamertag

    @property
    def gamertag(self) -> str:
        """Returns player gamertag"""
        return self._gamertag

    @property
    def uno(self) -> str:
        """Returns player uno"""
        return self._uno

    @uno.setter
    def uno(self, val: str):
        """Set Uno value"""
        self._uno = val

    @property
    def rebirth(self) -> dict:
        """Returns a dict of all mode stats for Rebirth"""
        return self._rebirth_stats

    @property
    def verdansk(self) -> dict:
        """Returns a dict of all mode stats for Verdansk"""
        return self._verdansk_stats

    @property
    def caldera(self) -> dict:
        """Returns a dict of all mode stats for Caldera"""
        return self._caldera_stats

    @property
    def rebirth_time(self) -> dict:
        """Returns a dict of time played for Rebirth"""
        return self._rebirth_time

    @property
    def verdansk_time(self) -> dict:
        """Returns a dict of time played for Verdansk"""
        return self._verdansk_time

    @property
    def caldera_time(self) -> dict:
        """Returns a dict of time played for Caldera"""
        return self._caldera_time

    @property
    def weapon_data(self) -> dict:
        """Returns weapon data"""
        return self._weapon_data


@dataclass
class Squad:
    """
    Calculate stats for all maps/modes for each squad memeber.

    :param squad_lst: List of gamertags. Include your gamertag in the list.
    :type squad_lst: List[str]
    :param original_df: Original DataFrame for stats to be calculated from.
    :type original_df: pd.DataFrame
    :param uno_name_dic: A dict of all gamertags and respective unos.
    :type uno_name_dic: dict
    :example:
        >>> from warzone.credentials import user_inputs
        >>> from warzone.user import User
        >>> from warzone.squad import Squad
        >>> _User = User(info=user_inputs)
        >>> _Squad = Squad(squad_lst=_User.squad_lst, original_df=cod.our_df, uno_name_dic=cod.name_uno_dict)
    :note: This will calculate and return the stats for all squad members.

    """
    squad_lst: List[str]
    original_df: pd.DataFrame
    uno_name_dic: Dict[str, str]

    def __init__(self, squad_lst, original_df, uno_name_dic):
        self._squad_stats = {teammate: Person(gamertag=teammate,
                                              original_df=original_df,
                                              uno=uno_name_dic[teammate]) for teammate in squad_lst}
        lst = []
        for teammate in squad_lst:
            for _mode in ['royale', 'resurgence']:
                for _team_size in ['solo', 'duo', 'trio', 'quad']:
                    self._squad_stats[teammate].rebirth[_mode][_team_size].stats['map'] = 'rebirth'
                    self._squad_stats[teammate].verdansk[_mode][_team_size].stats['map'] = 'verdansk'
                    self._squad_stats[teammate].caldera[_mode][_team_size].stats['map'] = 'caldera'
                    self._squad_stats[teammate].rebirth[_mode][_team_size].stats['mode'] = _mode
                    self._squad_stats[teammate].verdansk[_mode][_team_size].stats['mode'] = _mode
                    self._squad_stats[teammate].caldera[_mode][_team_size].stats['mode'] = _mode
                    self._squad_stats[teammate].rebirth[_mode][_team_size].stats['teamSize'] = _team_size
                    self._squad_stats[teammate].verdansk[_mode][_team_size].stats['teamSize'] = _team_size
                    self._squad_stats[teammate].caldera[_mode][_team_size].stats['teamSize'] = _team_size
                    self._squad_stats[teammate].rebirth[_mode][_team_size].stats['username'] = teammate
                    self._squad_stats[teammate].verdansk[_mode][_team_size].stats['username'] = teammate
                    self._squad_stats[teammate].caldera[_mode][_team_size].stats['username'] = teammate
                    lst.append(self._squad_stats[teammate].rebirth[_mode][_team_size].stats)
                    lst.append(self._squad_stats[teammate].verdansk[_mode][_team_size].stats)
                    lst.append(self._squad_stats[teammate].caldera[_mode][_team_size].stats)
        self._squad_df = pd.DataFrame(lst).fillna(0.0).set_index(['username', 'map', 'mode', 'teamSize'])

    def __repr__(self):
        return 'Squad Stats'

    @property
    def squad_dic(self):
        """Returns the dict of results"""
        return self._squad_stats

    @property
    def squad_df(self) -> pd.DataFrame:
        """Returns the dict of results in DataFrame format"""
        return self._squad_df
