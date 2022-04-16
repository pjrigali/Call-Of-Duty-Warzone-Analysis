from typing import Optional, Dict, Union, List
import pandas as pd
import numpy as np
import datetime
import os
import random
from warzone.utils.gun_dictionary import gun_dict
from warzone.classes.game import Game
from warzone.classes.window import Window
from warzone.utils.lists import SUM_LST, MU_LST, MAX_LST, HACKER_LST
from pyjr.utils.tools.clean import _mtype
from pyjr.utils.tools.general import _unique_values
from pyjr.utils.tools.dataframe import slc


# User
# None

# Squad
# None

# Person
def get_indexes_for_player(original_df: pd.DataFrame, uno: str) -> tuple:
    """Get indexes where a player is present."""
    lst = original_df["uno"].tolist()
    return tuple([ind for ind, val in enumerate(lst) if val == uno])


def get_player_stats(df: pd.DataFrame, stat_type: str):
    """Get desired stats for a player."""
    if stat_type == "sum":
        return {col: df[col].sum() for col in SUM_LST}
    elif stat_type == "mean":
        return {col: df[col].mean() for col in MU_LST}
    elif stat_type == "median":
        return {col: df[col].median() for col in MU_LST}
    elif stat_type == "max":
        return {col: df[col].max() for col in MAX_LST}


def get_player_weapons(df: pd.DataFrame, stat_type: str):
    """Get player weapons."""
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


def get_player_time(df: pd.DataFrame, stat_type: str):
    """Get player time."""
    seconds = df['timePlayed']
    minutes = seconds / 60
    hours = minutes / 60
    days = hours / 24

    if stat_type == "sum":
        return {"seconds": seconds.sum(), "minutes": minutes.sum(), "hours": hours.sum(), "days": days.sum()}
    elif stat_type == "mean":
        return {"seconds": seconds.mean(), "minutes": minutes.mean(), "hours": hours.mean(), "days": days.mean()}
    elif stat_type == "median":
        return {"seconds": seconds.median(), "minutes": minutes.median(), "hours": hours.median(),
                "days": days.median()}
    elif stat_type == "max":
        return {"seconds": seconds.max(), "minutes": minutes.max(), "hours": hours.max(), "days": days.max()}


# CallofDuty
def _build_from_json(path: str, save: bool = False) -> pd.DataFrame:
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


def streamer_mode_whole(_user_class, data: pd.DataFrame) -> None:
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


def streamer_mode_gamertags(_user) -> None:
    """Hides gamertags"""
    temp_lst = ['friend_gamertag_' + str(i + 1) for i, j in enumerate(_user.squad_lst)]
    _user.squad_lst = temp_lst
    _user.gamertag = temp_lst[0]


def streamer_mode_uno(_user, _squad) -> None:
    """Hides unos"""
    length, count = (20 - len(str(len(_user.squad_lst)))), 1
    for i in _user.squad_lst:
        val = '0' * length + str(count)
        _squad.squad_stats[i].uno = val
        count += 1


def _find_types(df: pd.DataFrame, keep_bool: bool = False):
    """Find best type to use to shrink dataframe."""
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


def _load_cols_dtypes(repo: str, from_json: bool):
    """Load pre-saved dtypes."""
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
        df = _build_from_json(path=json_path, save=build_json)
    elif build_json is False and from_json is True:
        if reset_dtype:
            df = pd.read_csv(json_path + 'CSV\\' + next(os.walk(json_path + 'CSV'))[2][0], dtype=dtype_dic,
                             index_col='Unnamed: 0')
            pd.DataFrame(_find_types(df=df, keep_bool=False), index=[0]).to_csv(json_path + 'CSV\\' + "dtype_info.csv",
                                                                                header=True)
        else:
            included_cols, dtype_dic_new = _load_cols_dtypes(repo=json_path, from_json=from_json)
            df = pd.read_csv(json_path + 'CSV\\' + next(os.walk(json_path + 'CSV'))[2][0], dtype=dtype_dic_new,
                             usecols=included_cols)
    else:
        if reset_dtype:
            df = pd.read_csv(repo + file_name, dtype=dtype_dic, index_col='Unnamed: 0').drop_duplicates(keep='first')
            pd.DataFrame(_find_types(df=df, keep_bool=False), index=[0]).to_csv(repo + "dtype_info.csv", header=True)
        else:
            included_cols, dtype_dic_new = _load_cols_dtypes(repo=repo, from_json=from_json)
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


def uno_username_dict(data: pd.DataFrame) -> dict:
    """Return a dict {gamertag: uno, gamertag1: uno1, etc}"""
    if 'uno' not in data.columns:
        raise AttributeError('uno required in dataframe')
    if 'username' not in data.columns:
        raise AttributeError('username required in dataframe')
    comb_set = set(data['uno'] + '-splitpoint-' + data['username'])
    return {i.split('-splitpoint-')[1]: i.split('-splitpoint-')[0] for i in comb_set if isinstance(i, str)}


def get_hacker_probability(our_df: pd.DataFrame, other_df: pd.DataFrame) -> list:
    """Calculates a Hacker Probability based on how stats relate to player and their squad makes"""
    our_data_dic = {"royale": {}, "resurgence": {}}
    for _mode in ["royale", "resurgence"]:
        for team_size in ['solo', 'duo', 'trio', 'quad']:
            our_data_n = our_df[(our_df['mode'] == _mode) & (our_df['teamSize'] == team_size)]
            our_data_dic[_mode][team_size] = {col: our_data_n[col].fillna(0.0).mean() for col in HACKER_LST}

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

    # Get hacker name uno dic
    temp_dic = {}
    for i in hacker_uno_dic.keys():
        temp = slc(data, 'uno', i)['username'].unique().tolist()
        for name in temp:
            temp_dic[name] = i
    return temp_dic, our_df, other_df


# TSanalysis
def _get_sessions(data: pd.DataFrame, minutes: int = 60) -> Dict[int, pd.DataFrame]:
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


def _get_group_stats(args) -> Dict[str, pd.DataFrame]:
    """Returns a dict with calculations for a given session"""
    data, lower, upper, mu_lst, sum_lst, quantile_lst, median_lst = args
    group = data.groupby('startDateTime')
    t = group.last()[['difficulty', 'ourPlacement']]
    return {'mu': pd.concat([group.mean()[MU_LST].sort_index(ascending=True), t], axis=1),
            'sum': pd.concat([group.sum()[SUM_LST].sort_index(ascending=True), t], axis=1),
            'median': pd.concat([group.median()[median_lst].sort_index(ascending=True), t], axis=1),
            'lower': pd.concat([group.quantile(q=lower)[quantile_lst].sort_index(ascending=True), t], axis=1),
            'upper': pd.concat([group.quantile(q=upper)[quantile_lst].sort_index(ascending=True), t], axis=1),
            'raw': data}


def _build_sessions(our_df: pd.DataFrame, other_df: pd.DataFrame, session_type: str,
                    session_value: int = None):
    """Get sessions outside of windows."""
    if session_type == 'day':
        window_matchid_dic = _get_matchids_day(df=our_df)
    elif session_type == 'game':
        window_matchid_dic = _get_matchids_game(df=our_df, session_value=session_value)
    elif session_type == 'event':
        window_matchid_dic = _get_matchids_event(df=our_df, session_value=session_value)
    elif session_type == 'session':
        window_matchid_dic = _get_matchids_session(df=our_df, session_value=session_value)
    else:
        raise AttributeError('Session_type must be {day, game, event, session}')
    our_ind_dic, other_ind_dic = _get_indexes(our_df=our_df, other_df=other_df, win_mat_dic=window_matchid_dic)
    our_df_dic, other_df_dic = _get_dfs(our_df=our_df, other_df=other_df, our_dic=our_ind_dic, other_dic=other_ind_dic,
                                        cols=False)
    return our_df_dic, other_df_dic


# DocumentFilter
def _check_empty(data: pd.DataFrame, return_empty: bool = True) -> pd.DataFrame:
    """Checks if the input dataframe is empty"""
    if return_empty:
        return data
    else:
        if data.empty:
            raise AttributeError('Based on input params, the dataframe will be empty.')
        else:
            return data


def _accept_list(data: pd.DataFrame, col: str, lst: Union[List[str], List[int]],
                 return_empty: bool = True) -> pd.DataFrame:
    """Handles a list of str's as an input"""
    if col not in data.columns:
        raise AttributeError(col + ' not included in the passed dataframe column list')
    else:
        lst_dic = {i: True for i in lst}
        data_lst = data[col].tolist()
        new_data = data.iloc[[i for i, j in enumerate(data_lst) if str(j) in lst_dic]]
        return _check_empty(data=new_data, return_empty=return_empty)


def _accept_str(data: pd.DataFrame, col: str, string: str, return_empty: bool = True) -> pd.DataFrame:
    """Handles a str input"""
    if col not in data.columns:
        raise AttributeError(col + ' not included in the passed dataframe column list')
    else:
        return _check_empty(data=data[data[col] == string], return_empty=return_empty)


def apply_filter(data: pd.DataFrame, col: str, val: Union[str, List[str], int, List[int]],
                 dic: Optional[Dict[str, str]] = None, return_empty: bool = True) -> pd.DataFrame:
    """Filters data based on input col and val"""
    if dic is None:
        if isinstance(val, str):
            return _accept_str(data=data, col=col, string=val, return_empty=return_empty)
        elif isinstance(val, list):
            return _accept_list(data=data, col=col, lst=val, return_empty=return_empty)
    else:
        if isinstance(val, str):
            return _accept_str(data=data, col=col, string=dic[val], return_empty=return_empty)
        elif isinstance(val, list):
            return _accept_list(data=data, col=col, lst=[dic[i] for i in val], return_empty=return_empty)


# Window
def _get_matchids_day(df: pd.DataFrame):
    """Get matchIDs for the windows based on day."""
    date_tup = _mtype(d=df['startDate'].unique(), dtype='tuple')
    dic = {}
    for ind, date in enumerate(date_tup):
        dic[ind] = _mtype(d=slc(df, 'startDate', date)['matchID'].unique(), dtype='tuple')
    return dic


def _get_matchids_game(df: pd.DataFrame, session_value: int) -> dict:
    """Get matchIDs for the windows based on game count."""
    id_tup = _mtype(d=df['matchID'].unique(), dtype='tuple')
    count, window = 0, 0
    id_lst = []
    dic = {}
    for _id in id_tup:
        id_lst.append(_id)
        count += 1
        if count == session_value:
            dic[window] = tuple(id_lst)
            count = 0
            window += 1
            id_lst = []
    return dic


def _get_matchids_event(df: pd.DataFrame, session_value: int):
    """Get matchIDs for the windows based on a teamPlacement number."""
    win_ids = _mtype(d=slc(df, 'teamPlacement', session_value)['matchID'].unique(), dtype='tuple')
    win_dic = {i: True for i in win_ids}
    window = 0
    dic = {}
    completed_dic = {}
    curr_dic = {}
    for ind, row in df.iterrows():
        if row['matchID'] in win_dic and row['matchID'] not in completed_dic:
            completed_dic[row['matchID']] = True
            dic[window] = _mtype(d=curr_dic.keys(), dtype='tuple')
            curr_dic = {row['matchID']: True}
            window += 1
        elif row['matchID'] in win_dic and row['matchID'] in completed_dic:
            pass
        else:
            curr_dic[row['matchID']] = True
    return {key - 1: val for key, val in dic.items() if key != 0}


def _get_matchids_session(df: pd.DataFrame, session_value: int):
    """Get matchIDs for the windows based on time in between games."""
    id_dic, ind_dic, count, past_game = {}, {}, 0, None
    for ind, row in df.iterrows():
        if past_game is None:
            past_game, id_dic[row['matchID']], ind_dic[count] = row['endDateTime'], True, [row['matchID']]
            continue
        elif row['matchID'] in id_dic:
            ind_dic[count].append(row['matchID'])
        else:
            if row['startDateTime'] - past_game <= datetime.timedelta(minutes=session_value):
                past_game, id_dic[row['matchID']] = row['endDateTime'], True
                ind_dic[count].append(row['matchID'])
            else:
                count += 1
                past_game, ind_dic[count] = row['endDateTime'], [row['matchID']]
    return {key: _unique_values(data=val, count=False) for key, val in ind_dic.items()}


def _get_indexes(our_df: pd.DataFrame, other_df: pd.DataFrame, win_mat_dic: dict):
    """Get indexes of matchIDs."""
    our_dic, other_dic = {}, {}
    for key, val in win_mat_dic.items():
        our_dic[key], other_dic[key] = [], []
        for _id in val:
            temp = _unique_values(data=list(slc(our_df, 'matchID', _id).index), count=False)
            for ind in temp:
                our_dic[key].append(ind)
            temp = _unique_values(data=list(slc(other_df, 'matchID', _id).index), count=False)
            for ind in temp:
                other_dic[key].append(ind)
    return our_dic, other_dic


def _get_dfs(our_df: pd.DataFrame, other_df: pd.DataFrame, our_dic: dict, other_dic: dict, cols: bool = True):
    """Get dataframes."""
    if cols:
        for key, val in our_dic.items():
            our_dic[key] = our_df.iloc[val][['matchID', 'startDateTime', 'endDateTime'] + SUM_LST]
        for key, val in other_dic.items():
            other_dic[key] = other_df.iloc[val][['matchID', 'startDateTime', 'endDateTime'] + SUM_LST]
    else:
        for key, val in our_dic.items():
            our_dic[key] = our_df.iloc[val]
        for key, val in other_dic.items():
            other_dic[key] = other_df.iloc[val]
    return our_dic, other_dic


def _build_windows(our_df: pd.DataFrame, other_df: pd.DataFrame, stat_type: str, session_type: str,
                   session_value: int = None) -> tuple:
    """Build Window class for each window."""
    if session_type == 'day':
        window_matchid_dic = _get_matchids_day(df=our_df)
    elif session_type == 'game':
        window_matchid_dic = _get_matchids_game(df=our_df, session_value=session_value)
    elif session_type == 'event':
        window_matchid_dic = _get_matchids_event(df=our_df, session_value=session_value)
    elif session_type == 'session':
        window_matchid_dic = _get_matchids_session(df=our_df, session_value=session_value)
    else:
        raise AttributeError('Session_type must be {day, game, event, session}')
    our_ind_dic, other_ind_dic = _get_indexes(our_df=our_df, other_df=other_df, win_mat_dic=window_matchid_dic)
    our_df_dic, other_df_dic = _get_dfs(our_df=our_df, other_df=other_df, our_dic=our_ind_dic, other_dic=other_ind_dic)
    win_lst = [Window(window=key, match_ids=val, team=our_df_dic[key], lobby=other_df_dic[key]) for key, val in window_matchid_dic.items()]
    for window in win_lst:
        game_lst, count = [], 0
        for _id in window.match_ids:
            game = Game(match_id=_id, position=count)
            game.add_team_lobby_stat(team_data=slc(window.team_df, 'matchID', _id),
                                     lobby_data=slc(window.lobby_df, 'matchID', _id), stat_type=stat_type)
            game_lst.append(game)
            count += 1
        window.games = tuple(game_lst)
    return tuple(win_lst)
