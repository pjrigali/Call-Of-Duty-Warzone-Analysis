"""User class object.

Usage:
 ./user.py

Author:
 Peter Rigali - 2021-08-30
"""
from typing import List, Tuple
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
        >>>     'squad': ['squadmate1', 'squadmate2', 'etc'],
        >>>     'file_name': 'Match_Data.csv',
        >>>     'hacker_file_name': 'hacker_df.csv',
        >>>     }
        >>> user = User(info=user_input_dict)
    :note: *None*

    """

    _file_name: str
    """File name of users data"""

    _hacker_file_name: str
    """File name of hacker data"""

    _repo: str
    """Directory location of data"""

    _json_repo: str
    """Directory location of json data"""

    _hacker_repo: str
    """Directory location of hacker json data"""

    _gamertag: str
    """Users gamertag"""

    _squad: Tuple[str]
    """List of gamertags"""

    def __init__(self, info: dict = None):

        if info is None:
            raise AttributeError('Need to pass an input dict')

        if info['file_name'] is None:
            raise AttributeError('Need to pass a file name')
        else:
            self._file_name: str = info['file_name']

        if info['hacker_file_name'] is None:
            raise AttributeError('Need to pass a file name')
        else:
            self._hacker_file_name: str = info['hacker_file_name']

        if info['repo'] is None:
            raise AttributeError('Need to pass a repo directory')
        else:
            self._repo: str = info['repo']

        if info['gamertag'] is None:
            raise AttributeError('Need to pass a gamertag')
        else:
            self._gamertag: str = info['gamertag']

        if info['squad'] is None:
            raise AttributeError('Need to pass a list of gamertags')
        else:
            self._squad: Tuple[str] = tuple(info['squad'])

        if info['json_repo'] is None:
            raise AttributeError('Need to pass a directory')
        else:
            self._json_repo: str = info['json_repo']

        if info['hacker_repo'] is None:
            raise AttributeError('Need to pass a directory')
        else:
            self._hacker_repo: str = info['hacker_repo']

        if self._gamertag not in self._squad:
            self._squad = tuple([self._gamertag] + list(self._squad))

    def __repr__(self):
        return self.gamertag

    @property
    def file_name(self) -> str:
        """Returns the file name of the users data"""
        return self._file_name

    @property
    def repo(self) -> str:
        """Returns the directory location of the users data"""
        return self._repo

    @property
    def json_repo(self) -> str:
        """Returns the directory location of the users json data"""
        return self._json_repo

    @property
    def hacker_repo(self) -> str:
        """Returns the directory location of the hacker json data"""
        return self._hacker_repo

    @property
    def gamertag(self) -> str:
        """Returns the users gamertag"""
        return self._gamertag

    @gamertag.setter
    def gamertag(self, val: str):
        """Set User Gamertag"""
        self._gamertag = val

    @property
    def squad_lst(self) -> Tuple[str]:
        """Returns the users squad gamertags as a list"""
        return self._squad

    @squad_lst.setter
    def squad_lst(self, lst: List[str]):
        """Set squad list"""
        self._squad = tuple(lst)
