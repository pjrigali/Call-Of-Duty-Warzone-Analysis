import pandas as pd
import numpy as np
from scipy import stats
from typing import List


def placement_descriptive_stats(our_data: pd.DataFrame,
                                other_data: pd.DataFrame,
                                col: str,
                                _map: str = None,
                                _name: str = None,
                                _dic: dict = None,
                                _internal: bool = False,
                                ) -> pd.DataFrame:
    
    if _name:
        our_data_n = our_data[our_data['uno'] == _dic[_name]]
    else:
        our_data_n = our_data.copy()
        
    if _internal:
        other_data_n = other_data[other_data['uno'] == _dic[_name]]
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
            
    return pd.DataFrame.from_dict(place_dic,
                                  orient='index',
                                  columns=['mean', 'std', 'var', 'max', 'min', 'count']).sort_index()


def first_top5_bottom_stats(data: pd.DataFrame,
                            col: str,
                            _map: str) -> pd.DataFrame:
    
    data = data.iloc[[i for i, j in enumerate(list(data['map'])) if _map in str(j)]]
    bottom = data[data['teamPlacement'] > 5][col]
    top_five = data[(data['teamPlacement'] <= 5) & (data['teamPlacement'] > 1)][col]
    top_one = data[data['teamPlacement'] == 1][col]

    base_dic = {'mu': [np.mean(top_one), np.mean(top_five), np.mean(bottom)],
                'std': [np.std(top_one), np.std(top_five), np.std(bottom)],
                'var': [np.var(top_one), np.var(top_five), np.var(bottom)],
                'max': [np.max(top_one), np.max(top_five), np.max(bottom)],
                'min': [np.min(top_one), np.min(top_five), np.min(bottom)]}
    
    base_df = pd.DataFrame.from_dict(base_dic, orient='columns')
    base_df.index = ['first', 'top five', 'bottom']
    
    return base_df


def bucket(data: pd.DataFrame, placement: list, col_lst: list, _map: str) -> pd.DataFrame:
    data = data.iloc[[i for i, j in enumerate(list(data['map'])) if _map in str(j)]]

    if len(placement) == 1:
        data = data[data['teamPlacement'] == placement]
    else:
        data = data[(data['teamPlacement'] < placement[1]) & (data['teamPlacement'] > placement[0])]

    id_lst = data['matchID'].unique()

    base_dic = {}
    for col in col_lst:
        base_dic[col + '_mu'] = []
        base_dic[col + '_std'] = []
        base_dic[col + '_var'] = []
        base_dic[col + '_max'] = []
        base_dic[col + '_min'] = []
        base_dic[col + '_skew'] = []
        base_dic[col + '_kurt'] = []

    for id in id_lst:
        temp = data[data['matchID'] == id]
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


def previous_next_placement(data: pd.DataFrame,
                            _map: str,
                            ) -> pd.DataFrame:
    
    data = data.iloc[[i for i, j in enumerate(list(data['map'])) if _map in str(j)]].reset_index(drop=True)
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


def weekly_stats(data: pd.DataFrame,
                 _map: str,
                 uno_dic: dict,
                 name: str
                 ) -> pd.DataFrame:
    
    data = data.iloc[[i for i, j in enumerate(list(data['map'])) if _map in str(j)]]
    data = data[data['uno'] == uno_dic[name]].sort_values('startDate').reset_index(drop=True)

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


def daily_stats(data: pd.DataFrame,
                _map: str,
                uno_dic: dict,
                name: str
                ) -> pd.DataFrame:

    data = data.iloc[[i for i, j in enumerate(list(data['map'])) if _map in str(j)]]
    data = data[data['uno'] == uno_dic[name]].sort_values('startDateTime').reset_index(drop=True)
    
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
                     sum_lst: List[str] = None
                     ) -> pd.DataFrame:

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


def squad_score_card(data: pd.DataFrame, usernames: List[str], username_dic: dict, _map: str) -> pd.DataFrame:
    base_df = data.iloc[[i for i, j in enumerate(list(data['map'])) if _map in str(j)]]
    col_lst = ['kdRatio', 'kills', 'deaths', 'damageDone', 'damageTaken', 'percentTimeMoving', 'distanceTraveled',
               'objectiveTeamWiped', 'objectiveReviver', 'missionsComplete', 'headshots', 'score', 'scorePerMinute']
    col_lst = col_lst + ['objectiveBrDownEnemyCircle' + str(i) for i in [6, 5, 4, 3, 2, 1]]

    people_dic = {}
    for person in usernames:
        temp_df = base_df[base_df['uno'] == username_dic[person]]
        temp_lst = [np.mean(temp_df[col]) for col in col_lst]
        head_ratio = np.mean(temp_df['headshots']) / np.mean(temp_df['kills'])
        max_kills = np.max(temp_df['kills'])
        max_deaths = np.max(temp_df['deaths'])
        max_streak = np.max(temp_df['longestStreak'])
        people_dic[person] = temp_lst + [head_ratio] + [max_kills] + [max_deaths] + [max_streak]

    col_lst_n = col_lst + ['headshotRatio'] + ['maxKills'] + ['maxDeaths'] + ['longestStreak']
    people_df = pd.DataFrame.from_dict(people_dic, orient='columns')
    people_df.index = col_lst_n
    return people_df.fillna(0)
