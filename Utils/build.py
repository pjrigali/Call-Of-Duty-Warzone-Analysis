from typing import List
import pandas as pd
import numpy as np
import datetime
from Classes.user import User
from Classes.squad import Squad


def _sm_whole(_user_class: User, data: pd.DataFrame) -> None:
    count = 1
    length = (20 - len(str(len(_user_class.squad))))
    temp_username_lst, temp_uno_lst = list(data['username']), list(data['uno'])
    for i in _user_class.squad:
        # Correct Gamertag
        gamertag_val = 'friend_gamertag_' + str(count)
        temp_username_lst = [gamertag_val if i == j else j for j in temp_username_lst]
        # Correct Uno
        target_uno = list(data[data['username'] == i]['uno'])[0]
        uno_val = '0' * length + str(count)
        temp_uno_lst = [uno_val if target_uno == j else j for j in temp_uno_lst]
        count += 1
    data['username'] = temp_username_lst
    data['uno'] = temp_uno_lst


def _sm_gamertags(_user: User) -> None:
    """Hides gamertags"""
    temp_lst = ['friend_gamertag_' + str(i + 1) for i, j in enumerate(_user.squad)]
    _user.set_squad(temp_lst)
    _user.set_gamertag(val=temp_lst[0])


def _sm_unos(_user: User, _squad: Squad) -> None:
    """Hides unos"""
    length, count = (20 - len(str(len(_user.squad)))), 1
    for i in _user.squad:
        val = '0' * length + str(count)
        _squad.squad_dic[i].set_uno(val=val)
        count += 1


def _evaluate_df(file_name: str, repo: str) -> pd.DataFrame:
    """Loads, Cleans, and Builds a DataFrame"""

    df = pd.read_csv(repo + file_name, index_col='Unnamed: 0').drop_duplicates(keep='first')

    # Create startDateTime
    start_time_utc_lst = list(df['utcStartSeconds'])
    df['startDateTime'] = [datetime.datetime.fromtimestamp(i) for i in start_time_utc_lst]

    # Create endDateTime
    end_time_utc_lst = list(df['utcEndSeconds'])
    df['endDateTime'] = [datetime.datetime.fromtimestamp(i) for i in end_time_utc_lst]

    # Create weekday column
    day_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    start_time_timestamp_lst = list(df['startDateTime'])
    df['weekDay'] = [day_dic[i.weekday()] for i in start_time_timestamp_lst]

    # Create startDate and startTime columns
    df['startDate'] = [datetime.datetime.strftime(i, '%Y-%m-%d') for i in start_time_timestamp_lst]
    df['startTime'] = [datetime.datetime.strftime(i, '%H:%M:%S') for i in start_time_timestamp_lst]

    # Create placementPercent column
    df['placementPercent'] = (1 - df['teamPlacement'] / df['teamCount']).round(2)

    # Calculate headshot ratio
    headshot_lst = list(df['headshots'])
    kill_lst = list(df['kills'])
    ran = range(len(df))
    hs_r = [0.0 if headshot_lst[ind] == 0 or kill_lst[ind] == 0 else headshot_lst[ind] / kill_lst[ind] for ind in ran]
    df['headshotRatio'] = hs_r

    # Update map column for Verdansk and Rebirth (mp_d or mp_e)
    map_lst = list(df['map'])
    df['map'] = ['mp_e' if 'mp_e' in i else 'mp_d' for i in map_lst]

    # Fix Blown out Damage Taken
    large_damage_taken = list(df[df['damageTaken'] > 100000].index)
    for i in large_damage_taken:
        df.loc[i, 'damageTaken'] = df.loc[i, 'damageDone']

    # Convert columns to str
    weapon_col_lst = [i for i in df.columns if ('primaryWeapon_' in i) | ('secondaryWeapon_' in i)]
    cols_lst = ['map', 'mode', 'team', 'username', 'uno', 'matchID'] + weapon_col_lst
    for col in cols_lst:
        temp_col_lst = list(df[col])
        df[col] = [str(i) for i in temp_col_lst]

    # Update mode column
    mode_lst = list(df['mode'])
    temp_lst = []
    for val in mode_lst:
        if 'quad' in val:
            temp_lst.append('quad')
        elif 'trio' in val:
            temp_lst.append('trio')
        elif 'duo' in val:
            temp_lst.append('duo')
        elif 'solo' in val:
            temp_lst.append('solo')
        else:
            temp_lst.append('other')
    df['mode'] = temp_lst

    return df.sort_values('startDateTime', ascending=True).reset_index(drop=True)


def _get_match_id_set(data: pd.DataFrame) -> dict:
    """Return a dict {gamertag: uno, gamertag1: uno1, etc}"""

    comb_set = set(data['uno'] + '-splitpoint-' + data['username'])
    return {i.split('-splitpoint-')[1]: i.split('-splitpoint-')[0] for i in comb_set}


def _get_hacker_probability(our_df: pd.DataFrame, other_df: pd.DataFrame, name_uno_dict: dict,
                            squad_name_lst: List[str]) -> list:
    """Calculates a Hacker Probability based on how stats relate to player and their squad makes"""

    col_lst = ['headshots', 'kills', 'deaths', 'kdRatio', 'scorePerMinute', 'distanceTraveled',
               'objectiveBrKioskBuy', 'percentTimeMoving', 'longestStreak', 'damageDone', 'damageTaken',
               'missionsComplete', 'objectiveLastStandKill', 'objectiveBrDownEnemyCircle1',
               'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3', 'objectiveBrDownEnemyCircle4',
               'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6', 'objectiveTeamWiped',
               'objectiveReviver', 'headshotRatio', 'objectiveMunitionsBoxTeammateUsed',
               'objectiveBrCacheOpen', 'objectiveMedalScoreKillSsRadarDrone']

    # Build our Mu's for comparison.
    our_data_dic = {}
    squad_uno_lst = [name_uno_dict[i] for i in squad_name_lst]
    for _map in ['mp_e', 'mp_d']:
        our_data_n = our_df[our_df['map'] == _map]
        temp = our_data_n.set_index('uno').loc[squad_uno_lst].fillna(0.0)
        our_data_dic[_map] = {col: np.mean(temp[col]) for col in col_lst}

    col_dic = {'mp_d': {'above': ['deaths', 'objectiveBrKioskBuy', 'missionsComplete',
                                  'objectiveMedalScoreKillSsRadarDrone'],
                        'below': ['headshots', 'kills', 'kdRatio', 'scorePerMinute', 'percentTimeMoving',
                                  'longestStreak', 'damageDone', 'objectiveLastStandKill',
                                  'objectiveBrDownEnemyCircle1',
                                  'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3',
                                  'objectiveBrDownEnemyCircle4', 'objectiveBrDownEnemyCircle5',
                                  'objectiveBrDownEnemyCircle6', 'objectiveTeamWiped', 'headshotRatio',
                                  'objectiveBrCacheOpen']},
               'mp_e': {'above': ['deaths', 'objectiveBrKioskBuy', 'damageTaken', 'missionsComplete',
                                  'objectiveMunitionsBoxTeammateUsed', 'objectiveBrCacheOpen',
                                  'objectiveMedalScoreKillSsRadarDrone'],
                        'below': ['headshots', 'kills', 'kdRatio', 'scorePerMinute', 'distanceTraveled',
                                  'percentTimeMoving', 'longestStreak', 'damageDone', 'objectiveLastStandKill',
                                  'objectiveBrDownEnemyCircle1', 'objectiveBrDownEnemyCircle2',
                                  'objectiveBrDownEnemyCircle3', 'objectiveBrDownEnemyCircle5',
                                  'objectiveBrDownEnemyCircle6', 'objectiveTeamWiped', 'objectiveReviver',
                                  'headshotRatio']}}

    dic = {i: [] for i in list(other_df.index)}
    for _map in ['mp_e', 'mp_d']:
        t = other_df.iloc[list(np.where(other_df['map'] == _map)[0])].fillna(0.0)
        for direction in ['above', 'below']:
            for criteria in col_dic[_map][direction]:
                if direction == 'above':
                    tt = list(t[t[criteria] < our_data_dic[_map][criteria]].index)
                else:
                    tt = list(t[t[criteria] > our_data_dic[_map][criteria]].index)
                tt_dic = {i: True for i in tt}
                for key in list(t.index):
                    if key in tt_dic:
                        dic[key].append(1)
                    else:
                        dic[key].append(0)
    dic_values_lst = list(dic.values())
    return [np.mean(np.nan_to_num(i)) for i in dic_values_lst]


def _get_our_and_other_df(data: pd.DataFrame, _my_uno: str, name_uno_dict: dict, squad_name_lst: List[str]):
    """Returns two DataFrames. First is all data related to the player and their teammates,
       Second is everyone who is not a teammate"""

    base_lst = data['matchID'] + '-splitpoint-' + data['team']
    base_our_lst = data[data['uno'] == _my_uno]['matchID'] + '-splitpoint-' + data[data['uno'] == _my_uno]['team']
    our_lst = {i: True for i in base_our_lst}
    comb_dic = {i: True for i, j in enumerate(base_lst) if j in our_lst}
    other = [i for i in data.index if i not in comb_dic]
    our_df, other_df = data.iloc[list(comb_dic.keys())].copy(), data.iloc[other].copy()
    our_df['hackerProb'] = 0.0
    other_df['hackerProb'] = _get_hacker_probability(our_df=our_df, other_df=other_df, name_uno_dict=name_uno_dict,
                                                     squad_name_lst=squad_name_lst)
    return our_df, other_df
