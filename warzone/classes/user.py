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
    :example: *None*
    :note: *None*

    """

    __slots__ = ["file_name", "hacker_file_name", "repo", "json_repo", "hacker_repo", "gamertag", "squad_lst",
                 "favorite", "hacker_json_repo"]

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

        self.hacker_json_repo = None
        if info['hacker_json_repo']:
            self.hacker_json_repo = info['hacker_json_repo']

    def __repr__(self):
        return self.gamertag
