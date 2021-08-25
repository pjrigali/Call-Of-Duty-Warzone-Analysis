import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Optional, Union
from Utils.gun_dictionary import gun_dict
from Utils.outlier import _stack, outlier_hist, outlier_std, outlier_var, outlier_distance, outlier_knn
from Utils.outlier import outlier_cooks_distance, outlier_regression
from Classes.document_filter import DocumentFilter


def first_top5_bottom_stats(doc_filter: DocumentFilter, col_lst: Union[List[str], str]) -> pd.DataFrame:
    """
    Calculate mu, std, var, max, min, skew, kurt for all matches depending on teamPlacement.

    The intent is for a map_choice and mode_choice to be fed into the DocumentFilter.
    Does calculations for all matches, regardless of matchID.

    Parameters
    ----------
    doc_filter : DocumentFilter
        A DocumentFilter.
    col_lst: List[str] or str
        Column or columns to get stats for.

    Returns
    ----------
    pd.DataFrame

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
        base_df[col + '_mu'] = [np.mean(top_one), np.mean(top_five), np.mean(bottom)]
        base_df[col + '_std'] = [np.std(top_one), np.std(top_five), np.std(bottom)]
        base_df[col + '_var'] = [np.var(top_one), np.var(top_five), np.var(bottom)]
        base_df[col + '_max'] = [np.max(top_one), np.max(top_five), np.max(bottom)]
        base_df[col + '_min'] = [np.min(top_one), np.min(top_five), np.min(bottom)]
        base_df[col + '_skew'] = [stats.skew(top_one), stats.skew(top_five), stats.skew(bottom)]
        base_df[col + '_kurt'] = [stats.kurtosis(top_one), stats.kurtosis(top_five), stats.kurtosis(bottom)]
    base_df.index = ['first', cut_off, 'bottom']
    return base_df


def bucket_stats(doc_filter: DocumentFilter, placement: Union[List[int], int],
                 col_lst: Union[List[str], str]) -> pd.DataFrame:
    """
    Calculate mu, std, var, max, min, skew, kurt for all matches depending on teamPlacement.

    The intent is for a map_choice and mode_choice to be fed into the DocumentFilter.
    Does calculations for all matches, considering of matchID.

    Parameters
    ----------
    doc_filter : DocumentFilter
        A DocumentFilter.
    placement: List[int] or int
        teamPlacement value used to filter data. If two int's are provided, will filter within that range.
        First value should be the lower value. Example [0,6] will return top 5 placements.
    col_lst: List[str] or str
        Column or columns to get stats for.

    Returns
    ----------
    pd.DataFrame

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
        for name in ['_mu', '_std', '_var', '_max', '_min', '_skew', '_kurt']:
            base_dic[col + name] = []

    id_lst = doc_filter.unique_ids
    for match_id in id_lst:
        temp = data[data['matchID'] == match_id]
        for col in col_lst:
            tempn = temp[col]
            base_dic[col + '_mu'].append(np.mean(tempn))
            base_dic[col + '_std'].append(np.std(tempn))
            base_dic[col + '_var'].append(np.var(tempn))
            base_dic[col + '_max'].append(np.max(tempn))
            base_dic[col + '_min'].append(np.min(tempn))
            base_dic[col + '_skew'].append(stats.skew(tempn))
            base_dic[col + '_kurt'].append(stats.kurtosis(tempn))

    base_df = pd.DataFrame.from_dict(base_dic, orient='columns').fillna(0.0)
    base_df.index = id_lst
    return base_df


def previous_next_placement(doc_filter: DocumentFilter) -> pd.DataFrame:
    """
    Calculate mu teamPlacement before and after a teamPlacement.

    The intent is for a map_choice and mode_choice to be fed into the DocumentFilter.

    Parameters
    ----------
    doc_filter : DocumentFilter
        A DocumentFilter.

    Returns
    ----------
    pd.DataFrame

    """

    data = doc_filter.df
    placement_lst = data['teamPlacement'].unique()
    placement_dic = {}
    for place in placement_lst:
        temp = data[data['teamPlacement'] == place].index
        temp_lst_prev, temp_lst_next = [], []
        for prev in temp:
            if prev - 1 > 0:
                temp_lst_prev.append(data.iloc[prev - 1]['teamPlacement'])
            if prev + 1 < np.max(temp):
                temp_lst_next.append(data.iloc[prev + 1]['teamPlacement'])
        placement_dic[place] = [np.mean(temp_lst_prev), np.mean(temp_lst_next)]
    col_lst = ['previous placement', 'next placement']
    return pd.DataFrame.from_dict(placement_dic, orient='index', columns=col_lst).sort_index()


def match_difficulty(our_doc_filter: DocumentFilter, other_doc_filter: DocumentFilter,
                     mu_lst: Optional[List[str]] = None, sum_lst: Optional[List[str]] = None,
                     test: Optional[bool] = False) -> pd.DataFrame:
    """
    Calculate the relative match difficulty based on player and player squad stats.

    The intent is for a map_choice and mode_choice to be fed into both DocumentFilter's.

    Parameters
    ----------
    our_doc_filter : DocumentFilter
        A DocumentFilter with squad and player data only.
    other_doc_filter : DocumentFilter
        A DocumentFilter with all other players data.
    mu_lst : List[str]
        A list of columns to consider the mu.
    sum_lst : List[str]
        A list of columns to consider the sum.
    test : bool
        If True, will use all columns for the analysis.

    Returns
    ----------
    pd.DataFrame

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
            match_dic[i][col + '_mu'] = np.mean(temp[col])
        for col in col_lst_sum:
            match_dic[i][col + '_sum'] = np.sum(temp[col])
        match_dic[i]['ourPlacement'] = np.mean(our_df[our_df['matchID'] == i]['placementPercent'])
    match_dic_df = pd.DataFrame.from_dict(match_dic, orient='index')

    if test:
        match_dic_df_corr = match_dic_df.corr().sort_values('ourPlacement')
        return match_dic_df_corr
    else:
        match_dic_df_norm = pd.DataFrame(index=match_dic_df.index)
        for i in list(match_dic_df.columns):
            temp = list(match_dic_df[i].fillna(0))
            minn, maxn = np.min(temp), np.max(temp)
            match_dic_df_norm[i] = [(j - minn) / (maxn - minn) for j in temp]

        lst = list(match_dic_df_norm.sum(axis=1) - match_dic_df_norm['ourPlacement'])
        minn, maxn = np.min(lst), np.max(lst)
        diff = pd.DataFrame([(j - minn) / (maxn - minn) for j in lst], columns=['difficulty'], index=match_dic_df.index)
        diff['ourPlacement'] = match_dic_df['ourPlacement']
        return diff


def get_daily_hourly_weekday_stats(doc_filter: DocumentFilter) -> list:
    """
    Calculate kills, deaths, wins, top 5s or 10s, match count, and averagePlacement for every day, week, hour.

    The intent is for a map_choice and mode_choice to be fed into the DocumentFilter.

    Parameters
    ----------
    doc_filter : DocumentFilter
        A DocumentFilter.

    Returns
    ----------
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict

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
        _matches = set(df['matchID'])
        if len(_matches) > 0:
            wins, top_fives = [], []
            for match in _matches:
                place = int(np.mean(df[df['matchID'] == match]['teamPlacement']))
                if place == 1:
                    wins.append(1)
                elif (1 < place) & (place >= num):
                    top_fives.append(1)
                else:
                    wins.append(0)
                    top_fives.append(0)

            return [int(df['kills'].sum()), int(df['deaths'].sum()), np.sum(wins), np.sum(top_fives), len(_matches),
                    np.mean(df['teamPlacement'])]
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
                if (val == 'averagePlacement') | (val == 'kdRatio'):
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
    Calculate the Kills. deaths, assists, headshots, averagePlacement and count for each weapon.

    The intent is for a username to be fed into the DocumentFilter and this will return the information for
    that specific player.

    Parameters
    ----------
    doc_filter : DocumentFilter
        A DocumentFilter.

    Returns
    ----------
    pd.DataFrame

    """
    data = doc_filter.df.fillna(0.0)
    val = sum([1 if 'primaryWeaponAttachement' in col else 0 for col in data.columns])
    col_names = []
    gun_dic = {}
    for i in range(1, val):
        name = 'primaryWeaponAttachements_' + str(i)
        gun_dic[name] = [ind for ind, j in enumerate(data[name]) if j != 0 and j.count('none') == 0]
        col_names.append(['primaryWeapon_' + str(i), 'secondaryWeapon_' + str(i)])

    gun_dict_keys_lst = [i for i in list(gun_dict.keys()) if i != 'none' and i != 'nan']
    gun_dic_2 = {i: {'kills': 0,
                     'deaths': 0,
                     'assists': 0,
                     'headshots': 0,
                     'averagePlacement': [],
                     'weaponType': i.split('_')[1],
                     'count': 0} for i in gun_dict_keys_lst}

    for i, j in enumerate(gun_dic.keys()):
        temp_df = data.iloc[gun_dic[j]]
        for weapon_name in gun_dict_keys_lst:
            for weapon_col in col_names[i]:
                t = temp_df[temp_df[weapon_col] == weapon_name]
                if t.empty is False:
                    for k in ['kills', 'deaths', 'headshots', 'assists']:
                        gun_dic_2[weapon_name][k] += np.sum(t[k])
                    gun_dic_2[weapon_name]['count'] += len(t)

                    for val in t['placementPercent']:
                        gun_dic_2[weapon_name]['averagePlacement'].append(val)

    for weapon_name in gun_dict_keys_lst:
        gun_dic_2[weapon_name]['averagePlacement'] = np.mean(gun_dic_2[weapon_name]['averagePlacement'])

    base_df = pd.DataFrame.from_dict(gun_dic_2, orient='index').fillna(0.0)
    base_df.index = [gun_dict[i] for i in list(base_df.index)]
    return base_df


def find_hackers(doc_filter: DocumentFilter, y_column: str, col_lst: List[str], std: int = 3) -> List[int]:
    """
    Calculate hackers based on various Outlier detection methods.

    The intent is for a map_choice and mode_choice to be fed into the DocumentFilter.

    Parameters
    ----------
    doc_filter : DocumentFilter
        A DocumentFilter.
    y_column : str
        A column to consider for Outlier analysis.
    col_lst : List[str]
        A list of columns used for Outlier analysis
    std : int, default is 3.
        The sta to be considered for as a threshold.

    Returns
    ----------
    List[int]

    Returns an index of suspected hackers.

    """

    data = doc_filter.df
    y_n = np.array(data[y_column])
    ind = []
    for col in col_lst:
        x_n = np.array(data[col])
        x_y = _stack(x_n, y_n, False)
        analysis = [list(outlier_var(arr=x_n, per=0.95, plus=True)),
                    list(outlier_std(arr=x_n, _std=std, plus=True)),
                    list(outlier_distance(arr=x_y, _std=std, plus=True)),
            #         list(outlier_hist(arr=x_n, per=0.75)),
            #         list(outlier_knn(arr=x_y, plus=True)),
            #         list(outlier_cooks_distance(arr=x_y, return_df=False)),
            #         list(outlier_regression(arr=x_y, _std=std))
                    ]
        ind.append(sum(analysis, []))

    temp_dict = {i: 0 for i in set(sum(ind, []))}
    for i in sum(ind, []):
        temp_dict[i] += 1
    return [i for i in temp_dict.keys() if temp_dict[i] >= 3 * len(col_lst)]


def meta_weapons(doc_filter: DocumentFilter, top_5_or_10: Optional[bool] = False, top_1: Optional[bool] = False) -> List[pd.DataFrame]:
    """
    Calculate the most popular weapons. Map_choice is required in DocumentFilter if top_5_or_10 or top_1 is True.
    If Neither top_5_or_10 or top_1 are True, it will calculate based on all team placements.

    This will only include loadouts where all attachment slots are filled. This calculates based on a daily interval.

    Parameters
    ----------
    doc_filter : DocumentFilter
        A DocumentFilter.
    top_5_or_10 : bool, default is False.
        If True, will calculate using only the top 5 or 10 place teams.
    top_1: bool, default is False.
        If True, will calculate using only the 1st place or winning team.

    Returns
    ----------
    List[pd.DataFrame]

    The First DataFrame is filled with dict's {kills: 0, deaths: 0, count: 0}.
    The Second is the percent of the lobby using.
    """

    if top_5_or_10 is True or top_1 is True:
        if doc_filter.map_choice is None:
            raise AttributeError('Include a map_choice in the DocumentFilter')

    data = doc_filter.df

    if doc_filter.map_choice == 'mp_d':
        num = 10
    elif doc_filter.map_choice == 'mp_e':
        num = 5

    if top_5_or_10:
        data = data[data['teamPlacement'] <= num].fillna(0).reset_index(drop=True)
    elif top_1:
        data = data[data['teamPlacement'] <= 1].fillna(0).reset_index(drop=True)
    else:
        data = data.fillna(0).reset_index(drop=True)

    dates = list(data['startDate'].unique())
    val = sum([1 if 'primaryWeaponAttachement' in col else 0 for col in data.columns])
    col_names = []
    gun_dic = {}
    for i in range(1, val):
        name = 'primaryWeaponAttachements_' + str(i)
        gun_dic[name] = [ind for ind, j in enumerate(data[name]) if j != 0 and j.count('none') == 0]
        col_names.append(['primaryWeapon_' + str(i), 'secondaryWeapon_' + str(i)])

    gun_dic_keys1 = list(gun_dic.keys())
    gun_dict_keys_lst = [i for i in list(gun_dict.keys()) if i != 'none' and i != 'nan']
    date_gun_dic = {date: {i: {'kills': 0, 'deaths': 0, 'count': 0} for i in gun_dict_keys_lst} for date in dates}
    date_dic = {date: list(data[data['startDate'] == date].index) for date in dates}
    for date in dates:
        lst1 = date_dic[date]
        for i, j in enumerate(gun_dic_keys1):
            lst2 = gun_dic[j]
            temp = data.iloc[[x for x in lst1 if x in lst2]]
            if temp.empty is False:
                for weapon_name in gun_dict_keys_lst:
                    for weapon_col in col_names[i]:
                        t = temp[temp[weapon_col] == weapon_name]
                        if t.empty is False:
                            date_gun_dic[date][weapon_name]['kills'] += np.sum(t['kills'])
                            date_gun_dic[date][weapon_name]['deaths'] += np.sum(t['deaths'])
                            date_gun_dic[date][weapon_name]['count'] += len(t)

    # for date in dates:
    #     for i, j in enumerate(gun_dic_keys1):
    #         temp_df = data.iloc[gun_dic[j]]
    #         temp = temp_df[temp_df['startDate'] == date]
    #         if temp.empty is False:
    #             for weapon_name in gun_dict_keys_lst:
    #                 for weapon_col in col_names[i]:
    #                     t = temp[temp[weapon_col] == weapon_name]
    #                     if t.empty is False:
    #                         date_gun_dic[date][weapon_name]['kills'] += np.sum(t['kills'])
    #                         date_gun_dic[date][weapon_name]['deaths'] += np.sum(t['deaths'])
    #                         date_gun_dic[date][weapon_name]['count'] += len(t)

    base_df = pd.DataFrame.from_dict(date_gun_dic, orient='index').sort_index()
    base_df.columns = [gun_dict[i] for i in list(base_df.columns)]

    ind_lst = list(base_df.index)
    final_df = pd.DataFrame()
    for ind in ind_lst:
        row = base_df.loc[ind]
        count_lst = [row.loc[weapon_name]['count'] for weapon_name in base_df.columns]
        final_df = pd.concat([final_df, pd.DataFrame([count_lst], columns=base_df.columns, index=[ind])])
    final_df_sum = final_df.sum(axis=1)
    final_df_sum.loc[final_df_sum == 0] = 1
    final_df = final_df.div(final_df_sum, axis=0)
    return [base_df, final_df]
