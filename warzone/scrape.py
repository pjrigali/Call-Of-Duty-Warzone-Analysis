"""Functions for dealing with new data.

Usage:
 ./scrape.py

Author:
 Peter Rigali - 2021-08-30
"""
from typing import List
from dataclasses import dataclass
import requests
import json
import pandas as pd
import time
from collections.abc import MutableMapping
from warzone.call_of_duty import CallofDuty


def _check_str(lst: list) -> tuple:
    return tuple([val if isinstance(val, str) else str(val) for val in lst])


def _check_dups(old_ids, new_ids):
    old_dic = {val: True for val in old_ids}
    return tuple([val for val in new_ids if val not in old_dic])


def _json_handling(_id: str, repo: str, error_lst: List[str], dump: bool = False):
    url = "https://www.callofduty.com/api/papi-client/crm/cod/v2/title/mw/platform/battle/fullMatch/wz/" + _id + "/it"
    response = requests.get(url).json()
    if response['status'] == 'success':
        if dump:
            with open(repo + _id + '.json', 'w', encoding='utf8') as f:
                json.dump(response, f, ensure_ascii=False)
        else:
            return response['data']['allPlayers']
    else:
        error_lst.append(_id)


def _flatten_dict_gen(d, parent_key, sep):
    """Function used in multithreading"""
    for k, v in d.items():
        if k != 'awards' and k != 'loadout':
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                yield from _flatten_dict(v, new_key, sep=sep).items()
            else:
                yield new_key, v


def _flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '_'):
    """Function used in multithreading"""
    return dict(_flatten_dict_gen(d, parent_key, sep))


def _shrink_loadout(row):
    """Reduces loadouts to a tuple"""
    pw = ''.join(['PW' + loadout['primaryWeapon']['name'] for loadout in row])
    sw = ''.join(['SW' + loadout['secondaryWeapon']['name'] for loadout in row])
    return pw + '__' + sw


@dataclass
class Scrape:

    __slots__ = ["old_ids", "new_ids", "dump", "seconds", "last_match_date_time", "repo", "json_repo", "error_lst",
                 "returned_data"]

    def __init__(self, cod: CallofDuty, dump: bool = False, seconds: float = 0.5, new_ids: List[str] = None):
        self.old_ids = _check_str(lst=cod.whole_df['matchID'].unique().tolist())
        self.new_ids = new_ids
        self.dump = dump
        self.seconds = seconds
        self.last_match_date_time = cod.last_match_date_time
        self.repo = cod.User.repo
        self.error_lst = []
        self.returned_data = None

        self.json_repo = None
        if cod.User.json_repo is not None:
            self.json_repo = cod.User.json_repo
            self.dump = False

        self.repo = None
        if cod.User.repo is not None:
            self.repo = cod.User.repo
        else:
            raise AttributeError('Must pass a repo in User input dict.')

    def set_new_ids(self, id_lst: list) -> None:
        self.new_ids = _check_dups(old_ids=self.old_ids, new_ids=_check_str(lst=id_lst))

    def get_data(self):
        """Collects and returns/dumps json files"""
        if self.new_ids is None:
            raise AttributeError('Need to call set_new_ids method first')
        count = len(self.new_ids)
        if self.json_repo is not None and self.dump == True:
            for _id in self.new_ids:
                _json_handling(_id=_id, repo=self.json_repo, dump=self.dump, error_lst=self.error_lst)
                time.sleep(self.seconds)
                count -= 1
                print(str(count) + ' ...remaining matches')
        else:
            self.returned_data = []
            for _id in self.new_ids:
                data = _json_handling(_id=_id, repo=self.repo, dump=self.dump, error_lst=self.error_lst)
                data_n = (player for player in data)
                self.returned_data.append(tuple([_flatten_dict(i) for i in data_n]))
                time.sleep(self.seconds)
                count -= 1
                print(str(count) + ' ...remaining matches')
            self.returned_data = tuple(self.returned_data)

    def get_dataframe(self) -> pd.DataFrame:
        if self.returned_data is None:
            raise AttributeError('Can only return Dataframe if dump is false.')
        else:
            df = pd.DataFrame([k for i in self.returned_data for k in i])
            lst = df['player_loadouts'].tolist()
            df['player_loadouts'] = [_shrink_loadout(row=person) for person in lst]
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
        return df

    def __repr__(self):
        return "ScrapeData"

# Test
# from warzone.scrape import Scrape
# s = Scrape(cod=cod, dump=False, new_ids=['9096526001368539833', '13091349904587640510'])
# s.get_data()
# ss = s.get_dataframe()
# ss


# def connect_to_api(_id: str):
#     """
#
#     Connect to Call of Duty API.
#
#     :param _id: A matchID str.
#     :type _id: str
#     :return: A Json of lobby data related to specified matchID.
#     :rtype: Json
#     :example: *None*
#     :note: Connect to Cod API to receive lobby information.
#
#     """
#
#     url = "https://www.callofduty.com/api/papi-client/crm/cod/v2/title/mw/platform/battle/fullMatch/wz/{}/it".format(
#         _id)
#     response = requests.request("GET", url, headers={})
#     return json.loads(response.text)['data']['allPlayers']


# def clean_api_data(json_object) -> pd.DataFrame:
#     """
#
#     Cleans the JSON output from *connect_to_api*
#
#     :param json_object: Json object.
#     :type json_object: Json
#     :return: Match information in a table.
#     :rtype: pd.DataFrame
#     :example: *None*
#     :note: Takes a Json object related to a matchID and constructs a pd.DataFrame with all relevant information.
#     This will need to be saved(or concatenated to an existing csv) and
#     loaded through the _evaulate_df() to work properly in this model.
#
#     """
#
#     base = []
#     for team in json_object:
#         player_data = {key: team[key] for key in team.keys() if key not in ['playerStats', 'player']}
#
#         for key in team['playerStats'].keys():
#             player_data[key] = team['playerStats'][key]
#
#         for key in team['player'].keys():
#
#             if key == 'brMissionStats':
#                 for i in team['player']['brMissionStats'].keys():
#                     if i == 'missionStatsByType':
#                         for j in team['player']['brMissionStats'][i].keys():
#                             for k in team['player']['brMissionStats'][i][j].keys():
#                                 player_data[j + '_' + k] = team['player']['brMissionStats'][i][j][k]
#                     else:
#                         player_data[i] = team['player']['brMissionStats'][i]
#             elif key == 'loadout':
#                 count = 1
#                 for i in team['player']['loadout']:
#                     player_data['primaryWeapon_' + str(count)] = i['primaryWeapon']['name']
#                     player_data['primaryWeaponAttachements_' + str(count)] = [k['name'] for k in
#                                                                               i['primaryWeapon']['attachments']]
#                     player_data['secondaryWeapon_' + str(count)] = i['secondaryWeapon']['name']
#                     player_data['secondaryWeaponAttachements_' + str(count)] = [k['name'] for k in
#                                                                                 i['secondaryWeapon']['attachments']]
#                     player_data['perks_' + str(count)] = [k['name'] for k in i['perks']]
#                     player_data['extraPerks_' + str(count)] = [k['name'] for k in i['extraPerks']]
#                     player_data['killstreaks_' + str(count)] = [k['name'] for k in i['killstreaks']]
#                     player_data['tactical_' + str(count)] = i['tactical']['name']
#                     player_data['lethal_' + str(count)] = i['lethal']['name']
#                     count += 1
#             else:
#                 player_data[key] = team['player'][key]
#         base.append(player_data)
#     return pd.DataFrame(base)


# def connect_and_dump(_id: str, repo: str):
#     """
#
#     Connect to Call of Duty API.
#
#     :param _id: A matchID str.
#     :type _id: str
#     :return: A Json of lobby data related to specified matchID.
#     :rtype: Json
#     :example: *None*
#     :note: Connect to Cod API to receive lobby information.
#
#     """
#     url = "https://www.callofduty.com/api/papi-client/crm/cod/v2/title/mw/platform/battle/fullMatch/wz/" + _id + "/it"
#     data = requests.get(url).json()
#     with open(repo + _id + '.json', 'w', encoding='utf8') as f:
#         json.dump(data, f, ensure_ascii=False)



# lst = cod.our_df['matchID'].unique().tolist()[1882:]
# import json
# import requests
# from os import walk
# from collections.abc import MutableMapping
# import time

# count = len(lst)
# for _id in lst:
#     data = requests.get("https://www.callofduty.com/api/papi-client/crm/cod/v2/title/mw/platform/battle/fullMatch/wz/{}/it".format(_id)).json()
#     with open('C:\\Users\\Peter\\Desktop\\Personal\\11_Repository\\Call of Duty Related\\Data\\Personal Jsons\\' + _id + '.json', 'w', encoding ='utf8') as f:
#         json.dump(data, f, ensure_ascii=False)
#     count -= 1
#     print(count)
#     time.sleep(.35)


# path = 'C:\\Users\\Peter\\Desktop\\Personal\\11_Repository\\Call of Duty Related\\Data\\Hacker Jsons'
# m = next(walk(path))[2]

#Open hacker csvs and scrape api
# lst = []
# for i in m:
#     temp = pd.read_csv(path + '\\' + i, index_col='Unnamed: 0')['0'].tolist()
#     for _id in temp:
#         lst.append(_id)
# lst = set(lst)
# count = len(lst)
# finished = []
# for _id in lst:
#     new_id = str(_id)
#     data = requests.get("https://www.callofduty.com/api/papi-client/crm/cod/v2/title/mw/platform/battle/fullMatch/wz/{}/it".format(new_id)).json()
#     with open('C:\\Users\\Peter\\Desktop\\Personal\\11_Repository\\Call of Duty Related\\Data\\Hacker Jsons\\' + new_id + '.json', 'w', encoding ='utf8') as f:
#         json.dump(data, f, ensure_ascii=False)
#     finished.append(_id)
#     count -= 1
#     print(count)
#     time.sleep(.4)