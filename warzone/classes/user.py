"""User class object.

Usage:
 ./warzone/classes/user.py

Author:
 Peter Rigali - 2021-08-30
"""
from dataclasses import dataclass


@dataclass
class User:
    """

    Organizes the Users input data.

    :param info: User input dict.
    :type info: dict
    :example:
        >>> from warzone.user import User
        >>> user_input_dict = {
        >>>     'repo': 'location of saved data',
        >>>     'json_repo': 'location of saved data in single json format',
        >>>     'hacker_repo': 'location of saved hacker data',
        >>>     'gamertag': 'your Ganertag',
        >>>     'squad_lst': ['squadmate1', 'squadmate2', 'etc'],
        >>>     'file_name': 'Match_Data.csv',
        >>>     'hacker_file_name': 'hacker_df.csv',
        >>>     'favorite': {'fav_map': 'rebirth', 'fav_mode': 'resurgence', 'fav_team_size': 'quad'}
        >>>     }
        >>> user = User(info=user_input_dict)
    :note: *None*

    """

    __slots__ = ["file_name", "hacker_file_name", "repo", "json_repo", "hacker_repo", "gamertag", "squad_lst",
                 "favorite"]

    def __init__(self, info: dict = None):

        if info is None:
            raise AttributeError('Need to pass an input dict')

        if info['file_name'] is None:
            raise AttributeError('Need to pass a file name')
        else:
            self.file_name = info['file_name']

        if info['repo'] is None:
            raise AttributeError('Need to pass a repo directory')
        else:
            self.repo = info['repo']

        if info['gamertag'] is None:
            raise AttributeError('Need to pass a gamertag')
        else:
            self.gamertag = info['gamertag']

        if info['squad_lst'] is None:
            raise AttributeError('Need to pass a list of gamertags')
        else:
            self.squad_lst = tuple(info['squad_lst'])

        if self.gamertag not in self.squad_lst:
            self.squad_lst = tuple([self.gamertag] + list(self.squad_lst))

        self.json_repo = None
        if info['json_repo']:
            self.json_repo = info['json_repo']

        self.hacker_repo = None
        if info['hacker_repo']:
            self.hacker_repo = info['hacker_repo']

        self.hacker_file_name = None
        if info['hacker_file_name']:
            self.hacker_file_name = info['hacker_file_name']

        self.favorite = None
        if info['favorite']:
            self.favorite = info['favorite']

    def __repr__(self):
        return self.gamertag
