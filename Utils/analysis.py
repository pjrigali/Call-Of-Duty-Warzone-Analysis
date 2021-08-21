import pandas as pd
import numpy as np
from scipy import stats
from typing import List

from Utils.gun_dictionary import gun_dict
from Utils.outlier import _stack, outlier_hist, outlier_std, outlier_var, outlier_distance, outlier_knn
from Utils.outlier import outlier_cooks_distance, outlier_regression

from Classes.document_filter import DocumentFilter


def placement_descriptive_stats(our_data: pd.DataFrame,
                                other_data: pd.DataFrame,
                                col: str,
                                _map: str = None,
                                username: str = None,
                                username_dic: dict = None,
                                _internal: bool = False,
                                ) -> pd.DataFrame:
    
    if username:
        our_data_n = our_data[our_data['uno'] == username_dic[username]]
    else:
        our_data_n = our_data.copy()
        
    if _internal:
        other_data_n = other_data[other_data['uno'] == username_dic[username]]
    else:
        other_data_n = other_data.copy()
    
    _our_data = our_data_n.iloc[[i for i, j in enumerate(list(our_data_n['map'])) if _map in str(j)]]
    _other_data = other_data_n.iloc[[i for i, j in enumerate(list(other_data_n['map'])) if _map in str(j)]]
    
    place_dic = {}
    for i in _our_data['teamPlacement'].unique():
        place_id_set = set(_our_data[_our_data['teamPlacement'] == i]['matchID'])
        
        if _internal:
            d = sum([list(_other_data[_other_data['matchID'] == s][col].fillna(0)) for s in place_id_set], [])
            _mean, _std, _var, _max, _min = np.mean(d), np.std(d), np.var(d), np.max(d), np.min(d)
            place_dic[i] = [_mean, _std, _var, _max, _min, len(place_id_set)]
        else:
            _mean, _std, _var, _max, _min = [], [], [], [], []
            for s in place_id_set:
                d = _other_data[_other_data['matchID'] == s][col].fillna(0)
                _mean.append(np.mean(d)), _std.append(np.std(d)), _var.append(np.var(d)), _max.append(np.max(d)), _min.append(np.min(d))
            place_dic[i] = [np.mean(_mean), np.mean(_std), np.mean(_var), np.mean(_max), np.mean(_min), len(place_id_set)]

    col_lst = ['mean', 'std', 'var', 'max', 'min', 'count']
    return pd.DataFrame.from_dict(place_dic, orient='index', columns=col_lst).sort_index()


def first_top5_bottom_stats(doc_filter: DocumentFilter, col: str) -> pd.DataFrame:
    data = doc_filter.df

    if col not in data.columns:
        raise AttributeError('Column given is not in data cols')

    if doc_filter.map_choice == 'mp_d':
        num = 10
        cut_off = 'top_10'
    else:
        num = 5
        cut_off = 'top_5'

    bottom = data[data['teamPlacement'] > num][col]
    top_five = data[(data['teamPlacement'] <= num) & (data['teamPlacement'] > 1)][col]
    top_one = data[data['teamPlacement'] == 1][col]

    base_dic = {'mu': [np.mean(top_one), np.mean(top_five), np.mean(bottom)],
                'std': [np.std(top_one), np.std(top_five), np.std(bottom)],
                'var': [np.var(top_one), np.var(top_five), np.var(bottom)],
                'max': [np.max(top_one), np.max(top_five), np.max(bottom)],
                'min': [np.min(top_one), np.min(top_five), np.min(bottom)]}
    
    base_df = pd.DataFrame.from_dict(base_dic, orient='columns')
    base_df.index = ['first', cut_off, 'bottom']
    return base_df


def bucket(doc_filter: DocumentFilter, placement: List[int], col_lst: List[str]) -> pd.DataFrame:
    data = doc_filter.df

    if placement[0] == placement[1]:
        data = data[data['teamPlacement'] == placement[0]]
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

    base_df = pd.DataFrame.from_dict(base_dic, orient='columns')
    base_df.index = id_lst
    return base_df


def previous_next_placement(doc_filter: DocumentFilter) -> pd.DataFrame:
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
    return pd.DataFrame.from_dict(placement_dic, orient='index', columns=['previous placement', 'next placement']).sort_index()


def weekly_stats(doc_filter: DocumentFilter) -> pd.DataFrame:
    data = doc_filter.df
    kills_lst, deaths_lst, top_five_lst, wins_lst, games_lst = [], [], [], [], []
    kills, deaths, top_fives, wins, games = 0, 0, 0, 0, 0
    for i, j in enumerate(data['weekDay']):
        if str(j) == 'Monday':
            if str(data['weekDay'].iloc[i - 1]) == 'Sunday':
                kills_lst.append(kills)
                deaths_lst.append(deaths)
                top_five_lst.append(top_fives)
                wins_lst.append(wins)
                games_lst.append(games)
                kills, deaths, top_fives, wins, games = 0, 0, 0, 0, 0
                
        kills += int(data['kills'].iloc[i])
        deaths += int(data['deaths'].iloc[i])
        games += 1
    
        placement = data['teamPlacement'].iloc[i]
        if (placement < 6) & (placement > 1):
            top_fives += 1
        if placement == 1:
            wins += 1

    base_df = pd.DataFrame()
    base_df['kills'] = kills_lst
    base_df['deaths'] = deaths_lst
    base_df['kd'] = base_df['kills'] / base_df['deaths']
    base_df['games'] = games_lst
    base_df['top 5s'] = top_five_lst
    base_df['wins'] = wins_lst
    base_df['win ratio'] = base_df['wins'] / base_df['games']
    base_df['kill ratio'] = base_df['kills'] / base_df['games']
    base_df['death ratio'] = base_df['deaths'] / base_df['games']
    return base_df


def daily_stats(doc_filter: DocumentFilter) -> pd.DataFrame:
    data = doc_filter.df
    kills_lst, deaths_lst, top_five_lst, wins_lst, games_lst = [], [], [], [], []
    kills, deaths, top_fives, wins, games = 0, 0, 0, 0, 0
    for i, j in enumerate(data['weekDay']):
        if str(data['weekDay'].iloc[i - 1]) != j:
            kills_lst.append(kills)
            deaths_lst.append(deaths)
            top_five_lst.append(top_fives)
            wins_lst.append(wins)
            games_lst.append(games)
            kills, deaths, top_fives, wins, games = 0, 0, 0, 0, 0
            
        kills += int(data['kills'].iloc[i])
        deaths += int(data['deaths'].iloc[i])
        games += 1
        
        placement = data['teamPlacement'].iloc[i]
        if (placement < 6) & (placement > 1):
            top_fives += 1
        if placement == 1:
            wins += 1
    
    base_df = pd.DataFrame()
    base_df['kills'] = kills_lst
    base_df['deaths'] = deaths_lst
    base_df['kd'] = base_df['kills'] / base_df['deaths']
    base_df['games'] = games_lst
    base_df['top 5s'] = top_five_lst
    base_df['wins'] = wins_lst
    base_df['win ratio'] = base_df['wins'] / base_df['games']
    base_df['kill ratio'] = base_df['kills'] / base_df['games']
    base_df['death ratio'] = base_df['deaths'] / base_df['games']
    base_df.index = data['startDate'].sort_values().unique()#[1:]
    
    return base_df


def match_difficulty(other_df: pd.DataFrame,
                     our_df: pd.DataFrame,
                     _map: str = 'mp_e',
                     test: bool = False,
                     mu_lst: List[str] = None,
                     sum_lst: List[str] = None) -> pd.DataFrame:

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

    data = other_df.iloc[[i for i, j in enumerate(list(other_df['map'])) if _map in str(j)]]
    match_dic = {i: {} for i in data['matchID']}
    for i in match_dic.keys():
        temp = data[data['matchID'] == i]
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


def get_daily_hourly_weekday_stats(doc_filter: DocumentFilter) -> List[pd.DataFrame]:
    data = doc_filter.df

    if doc_filter.map_choice == 'mp_d':
        num = 10
    else:
        num = 5

    def _stat_calc(df: pd.DataFrame):
        _matches = set(df['matchID'])
        if len(_matches) > 0:
            wins, top_fives = [], []
            for match in _matches:
                place = int(list(df[df['matchID'] == match]['teamPlacement'])[0])

                if place == 1:
                    wins.append(1)
                elif (1 < place) & (place >= num):
                    top_fives.append(1)
                else:
                    wins.append(0)
                    top_fives.append(0)

            return [int(df['kills'].sum()), int(df['deaths'].sum()), sum(wins), sum(top_fives), len(_matches),
                    df['teamPlacement'].mean()]
        else:
            return [0, 0, 0, 0, 0, 0]

    days_lst = data['startDate'].unique()
    col_lst = ['dailyKills', 'dailyDeaths', 'dailyWins', 'dailyTopFives', 'dailyMatchCount', 'dailyAverageTeamPlacement']
    days_dic = {day: _stat_calc(df=data[data['startDate'] == day]) for day in days_lst}
    daily_info = pd.DataFrame.from_dict(days_dic, orient='index', columns=col_lst)
    daily_info = daily_info[daily_info['dailyMatchCount'] > 0]
    daily_info['dailyKD'] = (daily_info['dailyKills'] / daily_info['dailyDeaths']).round(2)
    daily_info = daily_info.sort_index()

    hours = range(24)
    hours_lst = list(data['startTime'])
    col_lst = ['hourlyKills', 'hourlyDeaths', 'hourlyWins', 'hourlyTopFives', 'hourlyMatchCount',
               'hourlyAverageTeamPlacement']
    hourly_dic = {hour: _stat_calc(df=data.iloc[[i for i, j in enumerate(hours_lst) if hour == int(str(j).split(':')[0])]]) for hour in hours}
    hourly_info = pd.DataFrame.from_dict(hourly_dic, orient='index', columns=col_lst)
    hourly_info['hourlyKD'] = (hourly_info['hourlyKills'] / hourly_info['hourlyDeaths']).fillna(0).round(2)
    hourly_info.index = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00',
                         '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00',
                         '20:00', '21:00', '22:00', '23:00']

    weekdays_lst = data['weekDay'].unique()
    col_lst = ['weekDayKills', 'weekDayDeaths', 'weekDayWins', 'weekDayTopFives', 'weekDayMatchCount',
               'weekDayAverageTeamPlacement']
    week_dic = {weekday: _stat_calc(df=data[data['weekDay'] == weekday]) for weekday in weekdays_lst}
    weekday_info = pd.DataFrame.from_dict(week_dic, orient='index', columns=col_lst)
    weekday_info = weekday_info[weekday_info['weekDayMatchCount'] > 0]
    weekday_info['dailyKD'] = (weekday_info['weekDayKills'] / weekday_info['weekDayDeaths']).round(2)
    weekday_info.index = [str(i) for i in weekday_info.index]
    weekday_info = weekday_info.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # day_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    # hours = range(24)
    # dic = {}
    # for weekday in day_dic.values():
    #     dfn = data[data['weekDay'] == weekday].reset_index(drop=True)
    #     temp_dic = {}
    #     for _hour in hours:
    #         lst = []
    #         for i, j in enumerate(dfn['startDateTime']):
    #             if j.hour == _hour:
    #                 lst.append(dfn.loc[i, 'kills'])
    #         temp_dic[_hour] = sum(lst)
    #     dic[weekday] = temp_dic
    #
    # temp_df = pd.DataFrame.from_dict(dic)
    # temp_df.index = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00',
    #                  '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00',
    #                  '20:00', '21:00', '22:00', '23:00']

    return [daily_info, hourly_info, weekday_info]


def get_weapons(doc_filter: DocumentFilter) -> pd.DataFrame:
    data = doc_filter.df
    count = 0
    while True:
        if 'primaryWeaponAttachements_' + str(count) in data.columns:
            count += 1
        else:
            break

    col_names = []
    gun_dic = {}
    for i in range(1, count):
        name = 'primaryWeaponAttachements_' + str(i)
        gun_dic[name] = [ind for ind, j in enumerate(data[name]) if j != 0 and j.count('none') == 0]
        col_names.append(['primaryWeapon_' + str(i), 'secondaryWeapon_' + str(i)])

    gun_dict_keys_lst = [i for i in list(gun_dict.keys()) if i != 'none' and i != 'nan']
    gun_dic_2 = {i: {'kills': 0,
                     'deaths': 0,
                     'assists': 0,
                     'headshots': 0,
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

    base_df = pd.DataFrame.from_dict(gun_dic_2, orient='index')
    base_df.index = [gun_dict[i] for i in list(base_df.index)]
    return base_df


def find_hackers(doc_filter: DocumentFilter, y_column: str, col_lst: List[str], std: int = 3) -> np.ndarray:
    data = doc_filter.df
    y_n = np.array(data[y_column])
    ind = []
    for col in col_lst:
        x_n = np.array(data[col])
        x_y = _stack(x_n, y_n, False)
        analysis = [list(outlier_var(arr=x_n, _per=0.95, plus=True)),
                    list(outlier_std(arr=x_n, _std=std, plus=True)),
                    list(outlier_distance(arr=x_y, _std=std, plus=True)),
            #         list(outlier_hist(arr=x_n, _per=0.75)),
            #         list(outlier_knn(arr=x_y, plus=True)),
            #         list(outlier_cooks_distance(arr=x_y, return_df=False)),
            #         list(outlier_regression(arr=x_y, _std=std))
                    ]
        ind.append(sum(analysis, []))

    temp_dict = {i: 0 for i in set(sum(ind, []))}
    for i in sum(ind, []):
        temp_dict[i] += 1
    ind = np.array([i for i in temp_dict.keys() if temp_dict[i] >= 3 * len(col_lst)])
    return ind


def meta_weapons(doc_filter: DocumentFilter, top_5_or_10: bool = None, top_1: bool = None) -> List[pd.DataFrame]:
    data = doc_filter.df

    if doc_filter.map_choice == 'mp_d':
        num = 10
    else:
        num = 5

    if top_5_or_10:
        data = data[data['teamPlacement'] <= num].fillna(0).reset_index(drop=True)
    elif top_1:
        data = data[data['teamPlacement'] <= 1].fillna(0).reset_index(drop=True)
    else:
        data = data.fillna(0).reset_index(drop=True)

    dates = list(data['startDate'].unique())
    count = 0
    while True:
        if 'primaryWeaponAttachements_' + str(count) in data.columns:
            count += 1
        else:
            break

    col_names = []
    gun_dic = {}
    for i in range(1, count):
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
