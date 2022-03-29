"""One off functions for various analysis.

Usage:
 ./warzone/analysis.py

Author:
 Peter Rigali - 2021-08-30
"""
from typing import List, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats
from warzone.utils.gun_dictionary import gun_dict
from warzone.classes.document_filter import DocumentFilter


def first_top5_bottom_stats(doc_filter: DocumentFilter, col_lst: Union[List[str], str]) -> pd.DataFrame:
    """

    Calculate mu, std, var, max, min, skew, kurt for all matches depending on teamPlacement.
    Does calculations for all matches, regardless of matchID.

    :param doc_filter: Input DocumentFilter
    :type doc_filter: DocumentFilter
    :param col_lst: Input List of Columns to analyze.
    :type col_lst: List[str] or str
    :return: Stats, related to the items in col_lst, for winners, top 5 or 10, and bottom.
    :rtype: pd.DataFrame
    :example: *None*
    :note: If Rebirth is selected in the DocumentFilter, will return top 5. If Verdansk, top 10 is returned.
        The intent is for a map_choice and mode_choice to be fed into the DocumentFilter.

    """
    data = doc_filter.df

    if type(col_lst) == str:
        col_lst = [col_lst]

    for col in col_lst:
        if col not in data.columns:
            raise AttributeError(col + ' is not in data columns')

    if doc_filter.map_choice == 'mp_d':
        num = 10
        cut_off = 'top_10'
    else:
        num = 5
        cut_off = 'top_5'

    base_df = pd.DataFrame()
    for col in col_lst:
        bottom = data[data['teamPlacement'] > num][col]
        top_five = data[(data['teamPlacement'] <= num) & (data['teamPlacement'] > 1)][col]
        top_one = data[data['teamPlacement'] == 1][col]
        base_df[col + '_mu'] = [top_one.mean(), top_five.mean(), bottom.mean()]
        base_df[col + '_std'] = [top_one.std(), top_five.std(), bottom.std()]
        base_df[col + '_var'] = [top_one.var(), top_five.var(), bottom.var()]
        base_df[col + '_max'] = [top_one.max(), top_five.max(), bottom.max()]
        base_df[col + '_min'] = [top_one.min(), top_five.min(), bottom.min()]
        base_df[col + '_skew'] = [top_one.skew(), top_five.skew(), bottom.skew()]
        base_df[col + '_kurt'] = [top_one.kurt(), top_five.kurt(), bottom.kurt()]
    base_df.index = ['first', cut_off, 'bottom']
    return base_df


def bucket_stats(doc_filter: DocumentFilter, placement: Union[List[int], int],
                 col_lst: Union[List[str], str]) -> pd.DataFrame:
    """
    Calculate mu, std, var, max, min, skew, kurt for all matches depending on teamPlacement.
    Does calculations for all matches, considering of matchID.

    :param doc_filter: Input DocumentFilter.
    :type doc_filter: DocumentFilter
    :param placement: Target placement.
    :type placement: List[int] or int
    :param col_lst: Input List of Columns to analyze.
    :type col_lst: List[str] or str
    :return: Stats, related to the items in col_lst, for placement value.
    :rtype: pd.DataFrame
    :example: *None*
    :note: teamPlacement value used to filter data. If two int's are provided, will filter within that range.
        First value should be the lower value. Example [0,6] will return top 5 placements.
        The intent is for a map_choice and mode_choice to be fed into the DocumentFilter.
    """
    data = doc_filter.df

    if type(col_lst) == str:
        col_lst = [col_lst]

    if type(placement) == int:
        data = data[data['teamPlacement'] == placement]
    else:
        data = data[(data['teamPlacement'] < placement[1]) & (data['teamPlacement'] > placement[0])]

    base_dic = {}
    for col in col_lst:
        for name in ['_mu', '_std', '_var', '_max', '_min', '_skew', '_kurt', '_med']:
            base_dic[col + name] = []

    id_lst = doc_filter.unique_match_ids
    for match_id in id_lst:
        temp = data[data['matchID'] == match_id]
        for col in col_lst:
            tempn = temp[col]
            base_dic[col + '_mu'].append(tempn.mean())
            base_dic[col + '_std'].append(tempn.std())
            base_dic[col + '_var'].append(tempn.var())
            base_dic[col + '_max'].append(tempn.max())
            base_dic[col + '_min'].append(tempn.min())
            base_dic[col + '_skew'].append(tempn.skew())
            base_dic[col + '_kurt'].append(tempn.kurt())
            base_dic[col + '_med'].append(tempn.median())

    base_df = pd.DataFrame.from_dict(base_dic, orient='columns').fillna(0.0)
    base_df.index = id_lst
    return base_df


def previous_next_placement(doc_filter: DocumentFilter) -> pd.DataFrame:
    """
    Calculate mu teamPlacement before and after a teamPlacement.

    :param doc_filter: Input DocumentFilter.
    :type doc_filter: DocumentFilter
    :return: Previous and next expected placement based on current placement.
    :rtype: pd.DataFrame
    :example: *None*
    :note: The intent is for a map_choice and mode_choice to be fed into the DocumentFilter.
    """
    data = doc_filter.df
    placement_lst = data['teamPlacement'].unique()
    placement_dic = {}
    for place in placement_lst:
        temp = list(data[data['teamPlacement'] == place].index)
        temp_lst_prev, temp_lst_next = [], []
        for prev in temp:
            if prev - 1 > 0:
                temp_lst_prev.append(data.iloc[prev - 1]['teamPlacement'])
            if prev + 1 < max(temp):
                temp_lst_next.append(data.iloc[prev + 1]['teamPlacement'])
        placement_dic[place] = [np.mean(temp_lst_prev), np.mean(temp_lst_next)]
    col_lst = ['previous placement', 'next placement']
    return pd.DataFrame.from_dict(placement_dic, orient='index', columns=col_lst).sort_index()


def match_difficulty(our_doc_filter: DocumentFilter, other_doc_filter: DocumentFilter,
                     mu_lst: Optional[List[str]] = None, sum_lst: Optional[List[str]] = None,
                     test: Optional[bool] = False) -> pd.DataFrame:
    """

    Calculate the relative match difficulty based on player and player squad stats.

    :param our_doc_filter: A DocumentFilter with squad and player data only.
    :type our_doc_filter: DocumentFilter
    :param other_doc_filter: A DocumentFilter with all other players data.
    :type other_doc_filter: DocumentFilter
    :param mu_lst: A list of columns to consider the mu.
    :type mu_lst: List[str]
    :param sum_lst: A list of columns to consider the sum.
    :type sum_lst: List[str]
    :param test: If True, will use all columns for the analysis.
    :type test: bool
    :return: Match difficulty.
    :rtype: pd.DataFrame
    :example: *None*
    :note: The intent is for a map_choice and mode_choice to be fed into both DocumentFilter's.

    """
    our_df = our_doc_filter.df
    other_df = other_doc_filter.df
    if test:
        col_lst = ['duration', 'playerCount', 'teamCount', 'kills', 'medalXp', 'objectiveLastStandKill', 'matchXp',
                   'scoreXp', 'wallBangs', 'objectivePlunderCashBloodMoney', 'score', 'totalXp', 'headshots',
                   'assists',
                   'challengeXp', 'scorePerMinute', 'distanceTraveled', 'deaths', 'objectiveDestroyedEquipment',
                   'kdRatio', 'objectiveBrMissionPickupTablet', 'bonusXp', 'timePlayed', 'executions', 'nearmisses',
                   'objectiveBrCacheOpen', 'percentTimeMoving', 'miscXp', 'longestStreak', 'damageDone',
                   'damageTaken',
                   'missionsComplete', 'totalMissionXpEarned', 'totalMissionWeaponXpEarned', 'domination_weaponXp',
                   'domination_xp', 'domination_count', 'objectiveShieldDamage', 'objectiveBrKioskBuy',
                   'objectiveMunitionsBoxTeammateUsed', 'objectiveBrLootChopperBoxOpen', 'scavenger_weaponXp',
                   'scavenger_xp', 'scavenger_count', 'objectiveReviver', 'assassination_weaponXp',
                   'assassination_xp',
                   'assassination_count', 'objectiveBrC130BoxOpen', 'objectiveTeamWiped',
                   'objectiveDestroyedVehicleMedium', 'objectiveEmpedPlayer', 'objectivePerkMarkedTarget',
                   'objectiveDestroyedVehicleLight', 'objectiveTrophyDefense', 'teamSurvivalTime',
                   'objectiveBrDownEnemyCircle4', 'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle1',
                   'gulagDeaths', 'gulagKills', 'objectiveBrDownEnemyCircle3', 'timedrun_weaponXp', 'timedrun_xp',
                   'timedrun_count', 'objectiveMedalScoreSsKillTomaStrike', 'vip_weaponXp', 'vip_xp', 'vip_count',
                   'objectiveMedalScoreSsKillPrecisionAirstrike', 'objectiveBrDownEnemyCircle5',
                   'objectiveBrDownEnemyCircle6', 'objectiveMedalScoreKillSsRadarDrone',
                   'objectiveDestroyedVehicleHeavy', 'objectiveDestroyedTacInsert', 'blueprintextract_weaponXp',
                   'blueprintextract_xp', 'blueprintextract_count', 'objectiveBinocularsMarked',
                   'objectiveAssistDecoy',
                   'objectiveHack', 'objectiveManualFlareMissileRedirect', 'objectiveMedalScoreKillSsManualTurret',
                   'objectiveMedalScoreSsKillManualTurret', 'objectiveBinocularsAssist', 'objectiveEmpedVehicle',
                   'scavenger_soa_tower_weaponXp', 'scavenger_soa_tower_xp', 'scavenger_soa_tower_count',
                   'objectiveMedalScoreSsKillJuggernaut', 'result', 'objectiveWeaponDropTeammateUsed',
                   'objectiveShieldAssist', 'objectiveRadarDroneReconEnemyMarked', 'objectiveRadarDroneReconAssist',
                   'objectiveMedalScoreKillSsScramblerDrone', 'objectiveScrapAssist', 'objectiveTagCollected',
                   'objectiveBrGametypeBodycountFinalKill', 'objectiveTagDenied', 'placementPercent',
                   'teamPlacement']
        if mu_lst:
            col_lst_mu = mu_lst
        else:
            col_lst_mu = col_lst
        if sum_lst:
            col_lst_sum = sum_lst
        else:
            col_lst_sum = col_lst
    else:
        if mu_lst:
            col_lst_mu = mu_lst
        else:
            col_lst_mu = ['kdRatio', 'score', 'scorePerMinute', 'kills', 'longestStreak', 'headshots', 'damageDone',
                          'missionsComplete', 'distanceTraveled', 'objectiveTeamWiped', 'objectiveBrCacheOpen',
                          'objectiveLastStandKill']
        if sum_lst:
            col_lst_sum = sum_lst
        else:
            col_lst_sum = ['score', 'objectiveBrKioskBuy', 'objectiveBrDownEnemyCircle4',
                           'objectiveBrDownEnemyCircle3', 'objectiveTeamWiped', 'objectiveBrCacheOpen']

    match_dic = {i: {} for i in other_df['matchID']}
    for i in match_dic.keys():
        temp = other_df[other_df['matchID'] == i]
        for col in col_lst_mu:
            match_dic[i][col + '_mu'] = temp[col].mean()
        for col in col_lst_sum:
            match_dic[i][col + '_sum'] = temp[col].sum()
        match_dic[i]['ourPlacement'] = our_df[our_df['matchID'] == i]['placementPercent'].mean()
    match_dic_df = pd.DataFrame.from_dict(match_dic, orient='index')

    if test:
        match_dic_df_corr = match_dic_df.corr().sort_values('ourPlacement')
        return match_dic_df_corr
    else:
        match_dic_df_norm = pd.DataFrame(index=match_dic_df.index)
        for i in list(match_dic_df.columns):
            temp = match_dic_df[i].fillna(0)
            minn, maxn = temp.min(), temp.max()
            match_dic_df_norm[i] = [(j - minn) / (maxn - minn) for j in temp]

        lst = match_dic_df_norm.sum(axis=1) - match_dic_df_norm['ourPlacement']
        minn, maxn = lst.min(), lst.max()
        diff = pd.DataFrame(((j - minn) / (maxn - minn) for j in lst), columns=['difficulty'], index=match_dic_df.index)
        diff['ourPlacement'] = match_dic_df['ourPlacement']
        return diff


def get_daily_hourly_weekday_stats(doc_filter: DocumentFilter) -> list:
    """

    Calculate kills, deaths, wins, top 5s or 10s, match count, and averagePlacement for every day, week, hour.

    :param doc_filter: Input DocumentFilter.
    :type doc_filter: DocumentFilter
    :return: 3 pd.DataFrames and a dict
    :rtype:  *None*
    :example: *None*
    :note: The intent is for a map_choice and mode_choice to be fed into the DocumentFilter.

    """
    data = doc_filter.df
    hour_index_lst = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00',
                      '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00',
                      '20:00', '21:00', '22:00', '23:00']
    weekday_index_lst = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    if doc_filter.map_choice == 'mp_d':
        num = 10
    else:
        num = 5

    def _stat_calc(df: pd.DataFrame) -> list:
        """Stat calculator function"""
        _matches = set(df['matchID'])
        if len(_matches) > 0:
            wins, top_fives = [], []
            for match in _matches:
                place = int(df[df['matchID'] == match]['teamPlacement'].mean())
                if place == 1:
                    wins.append(1)
                elif 1 < place and place >= num:
                    top_fives.append(1)
                else:
                    wins.append(0)
                    top_fives.append(0)

            return [int(df['kills'].sum()), int(df['deaths'].sum()), sum(wins), sum(top_fives), len(_matches),
                    df['placementPercent'].mean()]
        else:
            return [0, 0, 0, 0, 0, 0]

    # Daily
    days_lst = data['startDate'].unique()
    if doc_filter.map_choice == 'mp_d':
        col_lst = ['dailyKills', 'dailyDeaths', 'dailyWins', 'dailyTop10s', 'dailyMatchCount',
                   'dailyAverageTeamPlacement']
    else:
        col_lst = ['dailyKills', 'dailyDeaths', 'dailyWins', 'dailyTop5s', 'dailyMatchCount',
                   'dailyAverageTeamPlacement']
    days_dic = {day: _stat_calc(df=data[data['startDate'] == day]) for day in days_lst}
    daily_info = pd.DataFrame.from_dict(days_dic, orient='index', columns=col_lst)
    daily_info = daily_info[daily_info['dailyMatchCount'] > 0]
    daily_info['dailyKD'] = (daily_info['dailyKills'] / daily_info['dailyDeaths']).round(2)
    daily_info = daily_info.sort_index()

    # Hourly
    hours = range(24)
    hours_lst = data['startDateTime']
    if doc_filter.map_choice == 'mp_d':
        col_lst = ['hourlyKills', 'hourlyDeaths', 'hourlyWins', 'hourlyTop10s', 'hourlyMatchCount',
                   'hourlyAverageTeamPlacement']
    else:
        col_lst = ['hourlyKills', 'hourlyDeaths', 'hourlyWins', 'hourlyTop5s', 'hourlyMatchCount',
                   'hourlyAverageTeamPlacement']

    hourly_dic = {h: _stat_calc(df=data.iloc[[i for i, j in enumerate(hours_lst) if h == j.hour]]) for h in hours}
    hourly_info = pd.DataFrame.from_dict(hourly_dic, orient='index', columns=col_lst)
    hourly_info['hourlyKD'] = (hourly_info['hourlyKills'] / hourly_info['hourlyDeaths']).fillna(0).round(2)
    hourly_info.index = hour_index_lst

    # Weekly
    weekdays_lst = data['weekDay'].unique()
    if doc_filter.map_choice == 'mp_d':
        col_lst = ['weekDayKills', 'weekDayDeaths', 'weekDayWins', 'weekDayTop10s', 'weekDayMatchCount',
                   'weekDayAverageTeamPlacement']
    else:
        col_lst = ['weekDayKills', 'weekDayDeaths', 'weekDayWins', 'weekDayTop5s', 'weekDayMatchCount',
                   'weekDayAverageTeamPlacement']
    week_dic = {weekday: _stat_calc(df=data[data['weekDay'] == weekday]) for weekday in weekdays_lst}
    weekday_info = pd.DataFrame.from_dict(week_dic, orient='index', columns=col_lst)
    weekday_info = weekday_info[weekday_info['weekDayMatchCount'] > 0]
    weekday_info['dailyKD'] = (weekday_info['weekDayKills'] / weekday_info['weekDayDeaths']).round(2)
    weekday_info.index = [str(i) for i in weekday_info.index]
    weekday_info = weekday_info.reindex(weekday_index_lst)

    hours = range(24)
    final_dic = {}
    for val in ['kills', 'deaths', 'kdRatio', 'wins', 'top_' + str(num), 'count', 'averagePlacement']:
        dic = {}
        for weekday in weekday_index_lst:
            dfn = data[data['weekDay'] == weekday].reset_index(drop=True)
            temp_dic = {}
            for _hour in hours:
                lst = []
                for i, j in enumerate(dfn['startDateTime']):
                    if j.hour == _hour:
                        if val == 'wins':
                            row = dfn.loc[i, 'teamPlacement']
                            if row == 1:
                                lst.append(1)
                        elif val == 'top_' + str(num):
                            row = dfn.loc[i, 'teamPlacement']
                            if (row > 1) & (row <= num):
                                lst.append(1)
                        elif val == 'count':
                            lst.append(1)
                        elif val == 'averagePlacement':
                            lst.append(dfn.loc[i, 'teamPlacement'])
                        else:
                            lst.append(dfn.loc[i, val])
                if val == 'averagePlacement' or val == 'kdRatio':
                    temp_dic[_hour] = np.mean(lst)
                else:
                    temp_dic[_hour] = sum(lst)
            dic[weekday] = temp_dic
        temp_df = pd.DataFrame.from_dict(dic)
        temp_df.index = hour_index_lst
        final_dic[val] = temp_df.fillna(0.0)

    return [daily_info, hourly_info, weekday_info, final_dic]


def get_weapons(doc_filter: DocumentFilter) -> pd.DataFrame:
    """
    Calculate the Kills, deaths, assists, headshots, averagePlacement and count for each weapon.

    :param doc_filter: Input DocumentFilter.
    :type doc_filter: DocumentFilter
    :return: A DataFrame with a players gun stats.
    :rtype: pd.DataFrame
    :example: *None*
    :note: The intent is for a username to be fed into the DocumentFilter and this will return the information for
        that specific player.

    """
    data = doc_filter.df

    # Get Weapons
    excluded_weapons = {"iw8_fists": True, "none": True, "nan": True}
    included_wzgunname_dic = {}
    included_gunname_dic = {}
    for wz_gun_name in gun_dict.keys():
        if wz_gun_name not in excluded_weapons:
            included_wzgunname_dic[wz_gun_name] = True
            included_gunname_dic[gun_dict[wz_gun_name]] = True

    # Get weapon indexes
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
    gun_stats_dic = {gun: {'kills': 0, 'deaths': 0, 'headshots': 0, 'assists': 0} for gun in list(gun_dict.values())}
    for key, value in gun_ind_dic.items():
        temp_df = data.iloc[list(value)]
        for col in ['kills', 'deaths', 'headshots', 'assists']:
            gun_stats_dic[key][col] = temp_df[col].sum()
        gun_stats_dic[key]['averagePlacementPercent'] = temp_df['placementPercent'].mean()
        gun_stats_dic[key]['count'] = len(temp_df)
        if gun_stats_dic[key]['kills'] != 0 and gun_stats_dic[key]['deaths'] != 0:
            gun_stats_dic[key]['kdRatio'] = gun_stats_dic[key]['kills'] / gun_stats_dic[key]['deaths']
        else:
            gun_stats_dic[key]['kdRatio'] = 0.0
        if '_' in weapon_type_dic[key]:
            gun_stats_dic[key]['weaponType'] = weapon_type_dic[key].split('_')[1]
        else:
            gun_stats_dic[key]['weaponType'] = 'None'
    final_df = pd.DataFrame.from_dict(gun_stats_dic, orient='index')
    return final_df


# def find_hackers(doc_filter: DocumentFilter, y_column: str, col_lst: List[str], std: int = 3) -> List[int]:
#     """
#
#     Calculate hackers based on various Outlier detection methods.
#
#     :param doc_filter: Input DocumentFilter.
#     :type doc_filter: DocumentFilter
#     :param y_column: A column to consider for Outlier analysis.
#     :type y_column: str
#     :param col_lst: A list of columns used for Outlier analysis.
#     :type col_lst: List[str]
#     :param std: The std to be considered for as a threshold, default is 3.
#     :type std: int
#     :return: Returns an index of suspected hackers.
#     :rtype: List[int]
#     :example: *None*
#     :note: The intent is for a map_choice and mode_choice to be fed into the DocumentFilter.
#
#     """
#     data = doc_filter.df
#     y_n = np.array(data[y_column])
#     ind = []
#     for col in col_lst:
#         x_n = np.array(data[col])
#         x_y = stack(x_n, y_n, False)
#         analysis = [list(outlier_var(arr=x_n, per=0.95, plus=True)),
#                     list(outlier_std(arr=x_n, _std=std, plus=True)),
#                     list(outlier_distance(arr=x_y, _std=std, plus=True)),
#             #         list(outlier_hist(arr=x_n, per=0.75)),
#             #         list(outlier_knn(arr=x_y, plus=True)),
#             #         list(outlier_cooks_distance(arr=x_y, return_df=False)),
#             #         list(outlier_regression(arr=x_y, _std=std))
#                     ]
#         ind.append(sum(analysis, []))
#
#     temp_dict = {i: 0 for i in set(sum(ind, []))}
#     for i in sum(ind, []):
#         temp_dict[i] += 1
#     return [i for i in temp_dict.keys() if temp_dict[i] >= 3 * len(col_lst)]


def meta_weapons(doc_filter: DocumentFilter, top_5_or_10: Optional[bool] = False, top_1: Optional[bool] = False,
                 col: Optional[str] = None, mu: Optional[bool] = None) -> pd.DataFrame:
    """

    Calculate the most popular weapons. Map_choice is required in DocumentFilter if top_5_or_10 or top_1 is True.
    If Neither top_5_or_10 or top_1 are True, it will calculate based on all team placements.
    This will only include loadouts where all attachment slots are filled. This calculates based on a daily interval.

    :param doc_filter: Input DocumentFilter.
    :type doc_filter: DocumentFilter
    :param top_5_or_10: If True, will calculate using only the top 5 or 10 place teams, default is False.
    :type top_5_or_10: bool
    :param top_1: If True, will calculate using only the 1st place or winning team.
    :type top_1: bool, default is False.
    :param col: If given will use a column as reference, default is None. None will count gun users per day.
    :type col: str
    :param mu: If True, will calculate using mean, default is sum.
    :type mu: bool
    :return: The First DataFrame is filled with dict's {kills: 0, deaths: 0, count: 0}.
        The Second is the percent of the lobby using.
    :rtype: pd.DataFrame
    :example: *None*
    :note: *None*

    """
    if top_5_or_10 is True or top_1 is True:
        if doc_filter.mode_choice is None:
            raise AttributeError('Include a mode_choice in the DocumentFilter')

    data = doc_filter.df

    if doc_filter.mode_choice == 'royale':
        num = 10
    else:
        num = 5

    if top_5_or_10:
        data = data[data['teamPlacement'] <= num].fillna(0).reset_index(drop=True)
    elif top_1:
        data = data[data['teamPlacement'] <= 1].fillna(0).reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    # Get Dates
    dates = data['startDate'].tolist()
    date_ind_dic = {date: set() for date in dates}
    for ind, date in enumerate(dates):
        date_ind_dic[date].add(ind)

    # Get Weapons
    excluded_weapons = {"iw8_fists": True, "none": True, "nan": True}
    included_wzgunname_dic = {}
    included_gunname_dic = {}
    for wz_gun_name in gun_dict.keys():
        if wz_gun_name not in excluded_weapons:
            included_wzgunname_dic[wz_gun_name] = True
            included_gunname_dic[gun_dict[wz_gun_name]] = True

    # Get Loadouts
    loadouts = data['loadouts'].tolist()

    # Get indexes for each weapon
    gun_ind_dic = {gun: set() for gun in included_gunname_dic.keys()}
    for ind, person in enumerate(loadouts):
        for loadout in person:
            primary, secondary = loadout[0], loadout[1]
            if primary != "iw8_fists" and secondary != "iw8_fists":
                if primary in included_wzgunname_dic:
                    gun_ind_dic[gun_dict[primary]].add(ind)
                if secondary in included_wzgunname_dic:
                    gun_ind_dic[gun_dict[secondary]].add(ind)

    # Count uses of weapon on specific days
    weapon_date_ind_dic = {date: {weapon_name: 0 for weapon_name in included_gunname_dic} for date in dates}
    for key, value in date_ind_dic.items():
        temp_df_index = list(data.iloc[list(value)].index)
        temp_df_index_dic = {ind: True for ind in temp_df_index}
        if col is None:
            for gun in gun_ind_dic.keys():
                for ind in gun_ind_dic[gun]:
                    if ind in temp_df_index_dic:
                        weapon_date_ind_dic[key][gun] += 1
        else:
            for gun in gun_ind_dic.keys():
                temp_ind = [ind for ind in gun_ind_dic[gun] if ind in temp_df_index_dic]
                weapon_date_ind_dic[key][gun] = sum(data.iloc[temp_ind][col])

    # Get percent use from total
    if mu:
        for key, value in weapon_date_ind_dic.items():
            total = sum(value.values())
            for key1, value1 in value.items():
                if value1 > 0:
                    weapon_date_ind_dic[key][key1] = value1 / total
                else:
                    weapon_date_ind_dic[key][key1] = 0.0
    final_df = pd.DataFrame.from_dict(weapon_date_ind_dic, orient='index')
    return final_df


def get_desired_kd(doc_filter: DocumentFilter, desired_kd: float, future_game_count: Optional[int] = 100,
                   max_kills_per_game: Optional[int] = None, min_deaths_per_game: Optional[int] = None,
                   use_dist: Optional[bool] = True, optimize: Optional[bool] = True) -> dict:
    """

    Calculates required kills per game, over a desired number of games to get a desired overall kd.

    :param doc_filter: Input DocumentFilter.
    :type doc_filter: DocumentFilter
    :param desired_kd: Desired kdRatio.
    :type desired_kd: float
    :param future_game_count: Number of future games to include in the simulation, default is 100. *Optional*
    :type future_game_count: int.
    :param max_kills_per_game: Upper limit on max kills per game, default is None. *Optional*
    :type max_kills_per_game: int.
    :param min_deaths_per_game: Lower limit on deaths per game, default is None. *Optional*
    :type min_deaths_per_game: int.
    :param use_dist: If True, will calculate kills and deaths using a normal distribution, default is True. *Optional*
    :type use_dist: bool.
    :param optimize: If True, will see if the desired kdRatio is possible in less games, default is True. *Optional*
    :type optimize: bool.
    :return: A dictionary with the following keys: {'Current kdRatio', 'Desired kdRatio', 'Games Required',
        'Resulting kdRatio', 'Required Kills Per Game', 'Required Deaths Per Game', 'Required Next kdRatio'}.
    :rtype: dict
    :example: *None*
    :note: The intent for this function is that the DocumentFilter has a map, mode, username, and username_dic.
        If no solution can be found the function raises and error.

    """
    if max_kills_per_game is None:
        max_kills_per_game = int(doc_filter.df['kills'].quantile(.977))
    if min_deaths_per_game is None:
        min_deaths_per_game = int(doc_filter.df['deaths'].quantile(.159))

    kills = doc_filter.df['kills'].sum()
    deaths = doc_filter.df['deaths'].sum()
    temp_dic = {'Current kdRatio': round(kills / deaths, 3), 'Desired kdRatio': round(desired_kd, 3),
                'Games Required': future_game_count, 'Resulting kdRatio': 0.0, 'Required Kills Per Game': 0,
                'Required Deaths Per Game': min_deaths_per_game, 'Required Next kdRatio': 0.0}
    for game_count in range(future_game_count, 1, -1):
        kills_per_game = 0
        if use_dist is False:
            for kill_count in range(1, max_kills_per_game):
                temp_kd = (kills + kill_count * game_count) / (deaths + min_deaths_per_game * game_count)
                if temp_kd >= desired_kd:
                    kills_per_game = kill_count
                    required_kd = (kill_count * game_count) / (min_deaths_per_game * game_count)
                    break
        else:
            kills_std = int(doc_filter.df['kills'].quantile(.159))
            if kills_std == 0:
                kills_std = 1
            else:
                kills_std = int(doc_filter.df['kills'].quantile(.50)) - kills_std
            deaths_mu = int(doc_filter.df['deaths'].quantile(.50))
            deaths_std = int(doc_filter.df['kills'].quantile(.159))
            if deaths_std < min_deaths_per_game:
                deaths_std = deaths_mu - min_deaths_per_game
            for kill_count in range(1, max_kills_per_game):
                kills_dist = stats.norm.rvs(kill_count, kills_std, size=game_count)
                temp_kills_lst = (int(i) for i in kills_dist)
                new_kills = sum([0 if i < 0 else i for i in temp_kills_lst])
                deaths_dist = stats.norm.rvs(deaths_mu, deaths_std, size=game_count)
                temp_deaths_lst = [int(i) for i in deaths_dist]
                new_deaths = sum([min_deaths_per_game if i < 0 else i for i in temp_deaths_lst])
                temp_kd = (kills + new_kills) / (deaths + new_deaths)
                if temp_kd >= desired_kd:
                    kills_per_game = kill_count
                    required_kd = new_kills / new_deaths
                    break

        if game_count <= temp_dic['Games Required'] and kills_per_game > 0 and required_kd >= desired_kd:
            temp_dic['Games Required'] = game_count
            temp_dic['Resulting kdRatio'] = round(temp_kd, 3)
            temp_dic['Required Kills Per Game'] = kills_per_game
            temp_dic['Required Next kdRatio'] = round(required_kd, 3)

        if optimize is False:
            break

    if temp_dic['Required Kills Per Game'] == 0:
        raise AttributeError('Either not enough future games or desired kd is too high')
    else:
        return temp_dic
