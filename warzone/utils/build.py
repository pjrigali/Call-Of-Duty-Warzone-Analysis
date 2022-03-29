"""Various internal functions used when building CallofDuty class objects.

Usage:
 ./warzone/utils/build.py

Author:
 Peter Rigali - 2021-08-30
"""
from typing import Optional
import pandas as pd
import numpy as np
import datetime
from warzone.classes.user import User
from warzone.classes.squad import Squad
import os
import random


def build_from_json(path: str, save: bool = False) -> pd.DataFrame:
    """Creates a Dataframe from json files"""
    import json
    from collections.abc import MutableMapping
    import concurrent.futures as cf

    # Keep all columns
    def _flatten_dict_gen(d, parent_key, sep):
        """Function used in multithreading"""
        for k, v in d.items():
            if k != 'awards' and k != 'loadout':
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, MutableMapping):
                    yield from flatten_dict(v, new_key, sep=sep).items()
                else:
                    yield new_key, v

    def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '_'):
        """Function used in multithreading"""
        return dict(_flatten_dict_gen(d, parent_key, sep))

    def load_files(args):
        """Function used in multithreading"""
        path, file = args
        with open(path + file, 'r', encoding="utf8") as f:
            data = json.load(f)['data']['allPlayers']
            data_n = (player for player in data)
            return (flatten_dict(i) for i in data_n)

    def load_files_main(path: str):
        """Multithreading function"""
        with cf.ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as p:
            files = (file for file in next(os.walk(path))[2] if '.json' in file)
            futures = (p.submit(load_files, (path, file)) for file in files)
            return [i for f in cf.as_completed(futures) for i in f.result()]

    df = pd.DataFrame(load_files_main(path=path))
    del MutableMapping

    def shrink_loadout(row):
        """Reduces loadouts to a tuple"""
        pw = ''.join(['PW' + loadout['primaryWeapon']['name'] for loadout in row])
        sw = ''.join(['SW' + loadout['secondaryWeapon']['name'] for loadout in row])
        return pw + '__' + sw

    lst = df['player_loadouts'].tolist()
    df['player_loadouts'] = [shrink_loadout(row=person) for person in lst]

    if save:
        if not os.path.exists(path + 'CSV'):
            os.makedirs(path + 'CSV')
        df.to_csv(path + 'CSV\\' + str(datetime.date.today()) +'.csv')

    return df


def sm_whole(_user_class: User, data: pd.DataFrame) -> None:
    """Hides gamertags and unos in whole dataset"""
    count = 1
    length = (20 - len(str(len(_user_class.squad_lst))))
    temp_username_lst, temp_uno_lst = data['username'].tolist(), data['uno'].tolist()
    for i in _user_class.squad_lst:
        gamertag_val = 'friend_gamertag_' + str(count)
        temp_username_lst = [gamertag_val if i == j else j for j in temp_username_lst]
        target_uno = list(data[data['username'] == i]['uno'])[0]
        uno_val = '0' * length + str(count)
        temp_uno_lst = [uno_val if target_uno == j else j for j in temp_uno_lst]
        count += 1
    data['username'] = temp_username_lst
    data['uno'] = temp_uno_lst


def sm_gamertags(_user: User) -> None:
    """Hides gamertags"""
    temp_lst = ['friend_gamertag_' + str(i + 1) for i, j in enumerate(_user.squad_lst)]
    _user.squad_lst = temp_lst
    _user.gamertag = temp_lst[0]


def sm_unos(_user: User, _squad: Squad) -> None:
    """Hides unos"""
    length, count = (20 - len(str(len(_user.squad_lst)))), 1
    for i in _user.squad_lst:
        val = '0' * length + str(count)
        _squad.squad_stats[i].uno = val
        count += 1


def find_types(df: pd.DataFrame, keep_bool: bool = False):
    type_dic = {"int": {"int8": (1, np.int8), "int16": (2, np.int16), "int32": (3, np.int32), "int64": (4, np.int64)},
                "uint": {"uint8": (1, np.uint8), "uint16": (2, np.uint16), "uint32": (3, np.uint32),
                         "uint64": (4, np.uint64)},
                "float": {"float16": (1, np.float16), "float32": (2, np.float32), "float64": (3, np.float64)}}
    class_name_dic = {str: "str", bool: "bool", np.bool_: "np.bool_"}
    for key, val in type_dic.items():
        for key1, val1 in val.items():
            class_name_dic[val1[1]] = key1
    col_type_dic = {}
    empty_cols = []
    other_type_cols = {}
    for col in df.columns:
        data = df[col].dropna().reset_index(drop=True)
        if data.empty:
            empty_cols.append(col)
            continue
        test_val = data[random.choice(range(len(data)))]
        if test_val is None:
            test_val = data[random.choice(range(len(data)))]
        if isinstance(test_val, str):
            col_type_dic[col] = str
        elif isinstance(test_val, (int, float, np.integer, np.float)):
            test_object = np.min_scalar_type(np.max(data))
            meta_type = test_object.name.split("t")[0] + "t"
            col_type_dic[col] = type_dic[meta_type][test_object.name][1]
        elif isinstance(test_val, (bool, np.bool_, np.bool)):
            if keep_bool:
                other_type_cols[col] = np.bool_
            else:
                empty_cols.append(col)
        else:
            other_type_cols[col] = type(test_val)
    final_dic = {i: "_empty_" for i in empty_cols}
    for key, val in col_type_dic.items():
        final_dic[key] = class_name_dic[val]
    return final_dic


def load_cols_dtypes(repo: str, from_json: bool):
    if from_json:
        repo1, repo2 = repo + 'CSV', repo + 'CSV\\' + "dtype_info.csv"
    else:
        repo1, repo2 = repo, repo + "dtype_info.csv"
    if "dtype_info.csv" in {str(i): True for i in next(os.walk(repo1))[2]}:
        type_df = pd.read_csv(repo2, index_col='Unnamed: 0')
        included_cols, new_dtype_dic = [], {}
        for col in type_df.columns:
            if type_df[col][0] != "_empty_":
                included_cols.append(col)
                new_dtype_dic[col] = type_df[col][0]
        return included_cols, new_dtype_dic
    else:
        return None, None


def evaluate_df(file_name: Optional[str] = None, repo: Optional[str] = None, json_path: Optional[str] = None,
                build_json: Optional[bool] = False, from_json: Optional[bool] = False,
                reset_dtype: bool = False) -> pd.DataFrame:
    """Loads, Cleans, and Builds a DataFrame"""
    dtype_dic = {'matchID': 'str',
                 'player_uno': 'str',
                 'player_username': 'str',
                 'player_team': 'str'}
    if build_json:
        df = build_from_json(path=json_path, save=build_json)
    elif build_json is False and from_json is True:
        if reset_dtype:
            df = pd.read_csv(json_path + 'CSV\\' + next(os.walk(json_path + 'CSV'))[2][0], dtype=dtype_dic,
                             index_col='Unnamed: 0')
            pd.DataFrame(find_types(df=df, keep_bool=False), index=[0]).to_csv(json_path + 'CSV\\' + "dtype_info.csv",
                                                                               header=True)
        else:
            included_cols, dtype_dic_new = load_cols_dtypes(repo=json_path, from_json=from_json)
            df = pd.read_csv(json_path + 'CSV\\' + next(os.walk(json_path + 'CSV'))[2][0], dtype=dtype_dic_new,
                             usecols=included_cols)
    else:
        if reset_dtype:
            df = pd.read_csv(repo + file_name, dtype=dtype_dic, index_col='Unnamed: 0').drop_duplicates(keep='first')
            pd.DataFrame(find_types(df=df, keep_bool=False), index=[0]).to_csv(repo + "dtype_info.csv", header=True)
        else:
            included_cols, dtype_dic_new = load_cols_dtypes(repo=repo, from_json=from_json)
            df = pd.read_csv(repo + file_name, dtype=dtype_dic_new, usecols=included_cols).drop_duplicates(keep='first')

    if build_json is True or from_json is True:
        lst = df['player_loadouts'].tolist()
        new_lst = []
        for loadout in lst:
            temp = []
            if len(loadout) > 2:
                pw, sw = loadout.split('__')[0].split('PW')[1:], loadout.split('__')[1].split('SW')[1:]
                for ind in range(len(pw)):
                    temp.append((pw[ind], sw[ind]))
            new_lst.append(temp)
        df['player_loadouts'] = new_lst

    # Fix Columns
    # temp_col_lst = df.columns.tolist()
    new_col_dic, new_col_lst = {}, []
    for col in df.columns:
        new_col = col
        for seperator in ['missionStatsByType_', 'brMissionStats_', 'playerStats_', 'player_']:
            if seperator in col:
                new_col = col.split(seperator)[1]
                break
        if new_col not in new_col_dic:
            new_col_dic[new_col] = True
            new_col_lst.append(new_col)
        else:
            new_col_dic[col] = True
            new_col_lst.append(col)
    df.columns = new_col_lst

    # Create startDateTime
    start_time_utc_lst = df['utcStartSeconds'].tolist()
    df['startDateTime'] = [datetime.datetime.fromtimestamp(i) for i in start_time_utc_lst]

    # Create endDateTime
    end_time_utc_lst = df['utcEndSeconds'].tolist()
    df['endDateTime'] = [datetime.datetime.fromtimestamp(i) for i in end_time_utc_lst]

    # Create weekday column
    day_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    start_time_timestamp_lst = df['startDateTime'].tolist()
    df['weekDay'] = [day_dic[i.weekday()] for i in start_time_timestamp_lst]

    # Create startDate and startTime columns
    df['startDate'] = [datetime.datetime.strftime(i, '%Y-%m-%d') for i in start_time_timestamp_lst]
    df['startTime'] = [datetime.datetime.strftime(i, '%H:%M:%S') for i in start_time_timestamp_lst]

    # Create placementPercent column
    df['placementPercent'] = (1 - df['teamPlacement'] / df['teamCount']).round(2)

    # Calculate headshot ratio
    headshot_lst = df['headshots'].tolist()
    kill_lst = df['kills'].tolist()
    ran = range(len(df))
    df['headshotRatio'] = [0.0 if headshot_lst[ind] == 0 or kill_lst[ind] == 0 else headshot_lst[ind] / kill_lst[ind] for ind in ran]
    # Update map column for Verdansk and Rebirth (mp_d or mp_e)
    # map_lst = df['map'].to_list()
    new_map_lst = []
    for row in df['map']:
        if "mp_d" in row:
            new_map_lst.append("verdansk")
            continue
        elif "mp_e" in row:
            new_map_lst.append("rebirth")
            continue
        elif "mp_w" in row:
            new_map_lst.append("caldera")
            continue
        else:
            new_map_lst.append("other")
    df['map'] = new_map_lst

    # Update mode and teamSize columns
    mode_lst = df['mode'].tolist()
    new_team_size_lst = []
    new_mode_lst = []
    resur_dic = {"br_rebirth_cal_res_royale": True, "br_rebirth_resurgence_mini": True, "br_rebirth_rbrthtrios": True,
                 "br_rebirth_rbrthduos": True, "br_rebirth_rbrthquad": True, "br_rebirth_resurgence_trios": True}
    royale_dic = {"br": True, "br_87": True, "br_25": True, "br_89": True, "br_71": True, "br_74": True, "br_88": True,
                  "br_brsolo": True, "br_brduos": True, "br_brtrios": True, "br_brquads": True, "br_br_real": True,
                  "br_86": True, "br_brthquad": True, "br_mini_miniroyale": True, "br_brbbsolo": True,
                  "br_brsolohwn": True, "br_mini_rebirth_mini_royale_quads": True,
                  "br_mini_rebirth_mini_royale_trios": True, "br_vg_royale_quads": True, "br_vg_royale_duos": True,
                  "br_vg_royale_solo": True, "br_vg_royale_trios": True}
    plunder_dic = {"br_dmz": True, "br_dmz_plndtrios": True, "br_dmz_38": True, "br_dmz_plunquad": True,
                   "br_dmz_76": True, "br_dmz_plndcndy": True, "br_dmz_85": True}
    solos_dic = {"br_87": True, "br_71": True, "br_brsolo": True, "br_brbbsolo": True, "br_vg_royale_solo": True,
                 "br_brsolohwn": True}
    duos_dic = {"br_88": True, "br_brduos": True, "br_vg_royale_duos": True, "br_dmz_85": True,
                "br_rebirth_rbrthduos": True, "br_dbd_iron_trials_duos": True}
    trios_dic = {"br_25": True, "br_74": True, "br_brtrios": True, "br_mini_rebirth_mini_royale_trios:": True,
                 "br_vg_royale_trios:": True, "br_dmz_plndtrios": True, "br_rebirth_rbrthtrios": True,
                 "br_rebirth_resurgence_trios:": True, "br_kingslayer_kingsltrios:": True, "br_brhwntrios": True,
                 "br_jugg_brtriojugr": True, "br_dmz_bldmnytrio": True, "br_dmz_bldmnytrio:": True}
    quads_dic = {"br_89": True, "br_brquads": True, "br_br_real": True, "br_86": True, "br_brthquad": True,
                 "br_mini_miniroyale": True, "br_mini_rebirth_mini_royale_quads": True, "br_vg_royale_quads": True,
                 "br_dmz_38": True, "br_dmz_plunquad": True, "br_dmz_76:": True, "br_dmz_plndcndy": True,
                 "br_rebirth_cal_res_royale": True, "br_rebirth_resurgence_mini": True, "br_rebirth_rbrthquad": True,
                 "br_truckwar_trwarsquads": True, "br_jugg_brquadjugr": True, "br_dbd_dbd": True, "br_77": True}
    mode_dic = {'resurgence': resur_dic, 'royale': royale_dic, 'plunder': plunder_dic}
    team_size_dic = {'solo': solos_dic, 'duo': duos_dic, 'trio': trios_dic, 'quad': quads_dic}

    def check_dic(val: str):
        """Converts the mode and teamSize columns"""
        temp_mode = 'other'
        for _mode in ['resurgence', 'royale', 'plunder']:
            if val in mode_dic[_mode]:
                temp_mode = _mode
                break
        new_mode_lst.append(temp_mode)

        temp_team_size = 'other'
        for _team_size in ['solo', 'duo', 'trio', 'quad']:
            if val in team_size_dic[_team_size]:
                temp_team_size = _team_size
                break
        new_team_size_lst.append(temp_team_size)

    for val in mode_lst:
        check_dic(val=val)

    df['teamSize'] = new_team_size_lst
    df['mode'] = new_mode_lst

    # Fix Blown out Damage Taken
    for i in list(df[df['damageTaken'] > 100000].index):
        df.loc[i, 'damageTaken'] = df.loc[i, 'damageDone']

    return df.sort_values('startDateTime', ascending=True).reset_index(drop=True)


def get_uno_username_dict(data: pd.DataFrame) -> dict:
    """Return a dict {gamertag: uno, gamertag1: uno1, etc}"""
    comb_set = set(data['uno'] + '-splitpoint-' + data['username'])
    return {i.split('-splitpoint-')[1]: i.split('-splitpoint-')[0] for i in comb_set if isinstance(i, str)}


def get_hacker_probability(our_df: pd.DataFrame, other_df: pd.DataFrame) -> list:
    """Calculates a Hacker Probability based on how stats relate to player and their squad makes"""
    col_lst = ['headshots', 'kills', 'deaths', 'kdRatio', 'scorePerMinute', 'distanceTraveled',
               'objectiveBrKioskBuy', 'percentTimeMoving', 'longestStreak', 'damageDone', 'damageTaken',
               'missionsComplete', 'objectiveLastStandKill', 'objectiveBrDownEnemyCircle1',
               'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3', 'objectiveBrDownEnemyCircle4',
               'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6', 'objectiveTeamWiped',
               'objectiveReviver', 'headshotRatio', 'objectiveMunitionsBoxTeammateUsed',
               'objectiveBrCacheOpen', 'objectiveMedalScoreKillSsRadarDrone']

    our_data_dic = {"royale": {}, "resurgence": {}}
    for _mode in ["royale", "resurgence"]:
        for team_size in ['solo', 'duo', 'trio', 'quad']:
            our_data_n = our_df[(our_df['mode'] == _mode) & (our_df['teamSize'] == team_size)]
            our_data_dic[_mode][team_size] = {col: our_data_n[col].fillna(0.0).mean() for col in col_lst}

    col_dic = {'royale': {'above': ['deaths', 'objectiveBrKioskBuy', 'missionsComplete',
                                    'objectiveMedalScoreKillSsRadarDrone'],
                          'below': ['headshots', 'kills', 'kdRatio', 'scorePerMinute', 'percentTimeMoving',
                                    'longestStreak', 'damageDone', 'objectiveLastStandKill',
                                    'objectiveBrDownEnemyCircle1', 'objectiveBrDownEnemyCircle2',
                                    'objectiveBrDownEnemyCircle3', 'objectiveBrDownEnemyCircle4',
                                    'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6', 'objectiveTeamWiped',
                                    'headshotRatio', 'objectiveBrCacheOpen']},
               'resurgence': {'above': ['deaths', 'objectiveBrKioskBuy', 'damageTaken', 'missionsComplete',
                                        'objectiveMunitionsBoxTeammateUsed', 'objectiveBrCacheOpen',
                                        'objectiveMedalScoreKillSsRadarDrone'],
                              'below': ['headshots', 'kills', 'kdRatio', 'scorePerMinute', 'distanceTraveled',
                                        'percentTimeMoving', 'longestStreak', 'damageDone', 'objectiveLastStandKill',
                                        'objectiveBrDownEnemyCircle1', 'objectiveBrDownEnemyCircle2',
                                        'objectiveBrDownEnemyCircle3', 'objectiveBrDownEnemyCircle5',
                                        'objectiveBrDownEnemyCircle6', 'objectiveTeamWiped', 'objectiveReviver',
                                        'headshotRatio']}}

    dic = {i: [] for i in list(other_df.index)}
    for _map in ['verdansk', 'rebirth', 'caldera']:
        for _mode in ['resurgence', 'royale']:
            for _team in ['solo', 'duo', 'trio', 'quad']:
                t = other_df.iloc[list(np.where((other_df['map'] == _map) & (other_df['mode'] == _mode) & (other_df['teamSize'] == _team))[0])].fillna(0.0)
                for direction in ['above', 'below']:
                    for criteria in col_dic[_mode][direction]:
                        if direction == 'above':
                            tt = list(t[t[criteria] < our_data_dic[_mode][team_size][criteria]].index)
                        else:
                            tt = list(t[t[criteria] > our_data_dic[_mode][team_size][criteria]].index)
                        tt_dic = {i: True for i in tt}
                        for key in list(t.index):
                            if key in tt_dic:
                                dic[key].append(1)
                            else:
                                dic[key].append(0)
    dic_values_lst = list(dic.values())
    ret = [sum(i) / len(i) if len(i) > 0 else 0.0 for i in dic_values_lst]
    return ret


def get_our_and_other_df(data: pd.DataFrame, _my_uno: str):
    """Returns two DataFrames. First is all data related to the player and their teammates,
       Second is everyone who is not a teammate"""

    base_lst = data['matchID'] + '-splitpoint-' + data['team']
    base_our_lst = data[data['uno'] == _my_uno]['matchID'] + '-splitpoint-' + data[data['uno'] == _my_uno]['team']
    our_lst = {i: True for i in base_our_lst}
    comb_dic = {i: True for i, j in enumerate(base_lst) if j in our_lst}
    other = [i for i in data.index if i not in comb_dic]
    our_df, other_df = data.iloc[list(comb_dic.keys())].copy(), data.iloc[other].copy()
    our_df['hackerProb'] = 0.0
    other_df['hackerProb'] = get_hacker_probability(our_df=our_df, other_df=other_df)
    return our_df, other_df


def get_hacker_and_other_df(data: pd.DataFrame, min_count: int = 5):
    """Similar to evaluate_df but used for the hacker data"""
    uno_lst = data['uno'].tolist()

    # Count
    dic, dic_count = {}, {}
    for _uno in uno_lst:
        if _uno in dic:
            dic_count[_uno] += 1
        else:
            dic[_uno] = True
            dic_count[_uno] = 1
    df = pd.DataFrame.from_dict(dic_count, orient='index')

    # Restrict
    n = max((np.quantile(df, .99), min_count))
    hacker_uno_dic = {_uno: True for _uno, val in dic_count.items() if val > n}
    temp_hacker_df = data.iloc[[i for i, j in enumerate(uno_lst) if str(j) in hacker_uno_dic]]
    hacker_id_team = (temp_hacker_df['matchID'] + '-splitpoint-' + temp_hacker_df['team']).unique().tolist()
    hacker_id_team_dic = {i: True for i in hacker_id_team}

    # Compare
    base_lst = (data['matchID'] + '-splitpoint-' + data['team']).tolist()
    comb_dic = {i: True for i, j in enumerate(base_lst) if j in hacker_id_team_dic}
    other = [i for i in data.index if i not in comb_dic]
    our_df, other_df = data.iloc[list(comb_dic.keys())].copy(), data.iloc[other].copy()
    return our_df, other_df
