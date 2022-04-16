"""CallofDuty class object.

Usage:
 ./warzone/classes/call_of_duty.py

Author:
 Peter Rigali - 2021-08-30
"""
from dataclasses import dataclass
from warzone.classes.user import User
from warzone.classes.squad import Squad
from warzone.utils.class_functions import streamer_mode_whole, streamer_mode_gamertags, evaluate_df
from warzone.utils.class_functions import uno_username_dict, get_our_and_other_df
from warzone.utils.gun_dictionary import gun_dict
from warzone.classes.hacker import Hacker


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
        >>> from warzone.classes.call_of_duty import CallofDuty
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

    __slots__ = ("User", "whole_df", "gun_dic", "last_match_date_time", "name_uno_dic", "my_uno", "our_df", "other_df",
                 "Squad", "hacker")

    def __init__(self,
                 user_input_dict: dict,
                 squad_data: bool = False,
                 hacker_data: bool = False,
                 streamer_mode: bool = False,
                 build_json: bool = False,
                 from_json: bool = False,
                 reset_dtype: bool = False):
        self.User = User(info=user_input_dict)
        self.whole_df = evaluate_df(file_name=self.User.file_name, repo=self.User.repo,
                                    json_path=self.User.json_repo, build_json=build_json,
                                    from_json=from_json, reset_dtype=reset_dtype)
        if streamer_mode:
            streamer_mode_whole(_user_class=self.User, data=self.whole_df)
        self.gun_dic = gun_dict
        self.last_match_date_time = self.whole_df['startDateTime'].tolist()[-1]
        self.name_uno_dic = uno_username_dict(data=self.whole_df)
        if streamer_mode:
            streamer_mode_gamertags(_user=self.User)
        self.my_uno = self.name_uno_dic[self.User.gamertag]
        self.our_df, self.other_df = get_our_and_other_df(data=self.whole_df, _my_uno=self.my_uno)
        self.hacker = None
        if hacker_data:
            self.hacker = Hacker(user=self.User, build_json=build_json, from_json=from_json, reset_dtype=reset_dtype)
        self.Squad = Squad(squad_lst=self.User.squad_lst, original_df=self.our_df, uno_name_dic=self.name_uno_dic,
                           build_all=squad_data, favorite=self.User.favorite)

    def __repr__(self):
        return 'Call of Duty'
