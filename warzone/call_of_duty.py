"""CallofDuty class object.

Usage:
 ./call_of_duty.py

Author:
 Peter Rigali - 2021-08-30
"""
from typing import Optional
import pandas as pd
from warzone.user import User
from warzone.squad import Squad
from warzone.build import sm_whole, sm_gamertags, evaluate_df, get_our_and_other_df, get_uno_username_dict
from warzone.build import get_hacker_and_other_df
from warzone.gun_dictionary import gun_dict
from dataclasses import dataclass


@dataclass
class CallofDuty:
    """

    Calculate stats for all maps/modes for each squad member.

    :param user_input_dict: A dict of user inputs.
    :type user_input_dict: dict
    :param squad_data: If True, will build the Squad class. default is True. *Optional*
    :type squad_data: bool
    :param hacker_data: This Requires a seperate csv with hacker data saved. This data can be collected by
        finding hackers after the fact and scraping there data from CodTracker, this can then be used to find
        hackers in other games. Default is False. *Optional*
    :type hacker_data: bool
    :param streamer_mode: If True, will hide User inputted Gamertag's and Uno's. default is False. *Optional*
    :type streamer_mode: bool
    :param build_json: If True, will build the data from a folder of jsons *Optional*
    :type build_json: bool
    :param from_json: If True will load the csv created from the json files. *Optional*
    :type from_json: bool
    :example:
        >>> from warzone.call_of_duty import CallofDuty
        >>> user_input_dict = {
        >>>     'repo': 'location of saved data',
        >>>     'json_repo': 'location of saved data in single json format',
        >>>     'hacker_repo': 'location of saved hacker data',
        >>>     'gamertag': 'your Ganertag',
        >>>     'squad': ['squadmate1', 'squadmate2', 'etc'],
        >>>     'file_name': 'Match_Data.csv',
        >>>     'hacker_file_name': 'hacker_df.csv',
        >>>     }
        >>> cod = CallofDuty(user_input_dict=user_input_dict, squad_data=True, hacker_data=False, streamer_mode=False)
    :note: This will calculate and build the CallofDuty class.

    """
    def __init__(self,
                 user_input_dict: dict,
                 squad_data: bool = True,
                 hacker_data: Optional[bool] = False,
                 streamer_mode: Optional[bool] = False,
                 build_json: Optional[bool] = False,
                 from_json: Optional[bool] = False):
        self._User = User(info=user_input_dict)
        self._whole: pd.DataFrame = evaluate_df(file_name=self._User.file_name, repo=self._User.repo,
                                                json_path=self._User.json_repo, build_json=build_json,
                                                from_json=from_json)

        if streamer_mode:
            sm_whole(_user_class=self._User, data=self._whole)

        self._gun_dic = gun_dict
        self._last_match_date_time = self._whole['startDateTime'].tolist()[-1]
        self._name_uno_dict = get_uno_username_dict(data=self._whole)

        if streamer_mode:
            sm_gamertags(_user=self._User)

        self._my_uno = self.name_uno_dict[self._User.gamertag]
        self._our_df, self._other_df = get_our_and_other_df(data=self._whole, _my_uno=self._my_uno,
                                                            squad_name_lst=self._User.squad_lst,
                                                            name_uno_dict=self._name_uno_dict)
        self._hacker_whole = None
        self._hacker_name_uno_dict = None
        self._hacker_df = None
        self._other_hacker_df = None
        if hacker_data:
            self._hacker_whole = evaluate_df(file_name=None, repo=self._User.repo, json_path=self._User.hacker_repo,
                                             build_json=build_json, from_json=from_json)
            self._hacker_name_uno_dict = get_uno_username_dict(data=self._hacker_whole)
            self._hacker_df, self._hacker_other_df = get_hacker_and_other_df(data=self._hacker_whole)

        self._Squad = None
        if squad_data:
            self._Squad = Squad(squad_lst=self._User.squad_lst, original_df=self.our_df, uno_name_dic=self.name_uno_dict)

    def __repr__(self):
        return 'Call of Duty'

    @property
    def whole(self) -> pd.DataFrame:
        """The unedited player matches DataFrame"""
        return self._whole

    @property
    def gun_dictionary(self) -> dict:
        """Returns a dict of gun names"""
        return self._gun_dic

    @property
    def last_match_date_time(self):
        """Returns a Timestamp of the latest game in the players data. Useful when scraping from Cod Tracker"""
        return self._last_match_date_time

    @property
    def name_uno_dict(self) -> dict:
        """Returns a dict of gamertags and respective unos"""
        return self._name_uno_dict

    @property
    def my_uno(self) -> str:
        """Returns the user uno value"""
        return self._my_uno

    @property
    def our_df(self) -> pd.DataFrame:
        """Returns a DataFrame of all data related to player and there teammates"""
        return self._our_df

    @property
    def other_df(self) -> pd.DataFrame:
        """Returns a DataFrame of all data related to other teams in a lobby"""
        return self._other_df

    @property
    def hacker_whole(self) -> pd.DataFrame:
        """If a hacker data is provided, will return just the hacker DataFrame"""
        return self._hacker_whole

    @property
    def hacker_df(self) -> pd.DataFrame:
        """Returns a DataFrame of all data related to hackers and there teammates"""
        return self._hacker_df

    @property
    def hacker_other_df(self) -> pd.DataFrame:
        """Returns a DataFrame of all data related to other teams in a lobby, from hacker data"""
        return self._hacker_other_df

    @property
    def hacker_name_uno_dict(self) -> dict:
        """If a hacker DataFrame is provided, will return the gamertags: unos for the hacker DataFrame"""
        return self._hacker_name_uno_dict

    @property
    def user(self) -> User:
        """Returns a User class object of related info to the user"""
        return self._User

    @property
    def squad(self) -> Squad:
        """Returns a Squad class object of stats related to the user squad mates"""
        return self._Squad
