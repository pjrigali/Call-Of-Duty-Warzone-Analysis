"""Game class object.

Usage:
 ./warzone/classes/game.py

Author:
 Peter Rigali - 2022-03-30
"""
from dataclasses import dataclass
import pandas as pd
from pyjr.classes.stat import Stat
from warzone.utils.lists import SUM_LST


@dataclass
class Game:
    """Builds a single game"""
    __slots__ = ['team', 'lobby', 'position', 'match_id']

    def __init__(self, position: int, match_id: str):
        self.match_id = match_id
        self.team = None
        self.lobby = None
        self.position = position

    def add_team_stat(self, data: pd.DataFrame, stat_type: str):
        """Add team stats"""
        stat = Stat(stat=stat_type, na='zero', dtype='float', empty=True)
        self.team = tuple([stat.get(data=data[col], q=None) for col in SUM_LST])
        return self

    def add_lobby_stat(self, data: pd.DataFrame, stat_type: str):
        """Add lobby stats"""
        stat = Stat(stat=stat_type, na='zero', dtype='float', empty=True)
        self.lobby = tuple([stat.get(data=data[col], q=None) for col in SUM_LST])
        return self

    def add_team_lobby_stat(self, team_data: pd.DataFrame, lobby_data: pd.DataFrame, stat_type: str):
        """Add both team and lobby stats"""
        stat = Stat(stat=stat_type, na='zero', dtype='float', empty=True)
        self.team = tuple([stat.get(data=team_data[col], q=None) for col in SUM_LST])
        self.lobby = tuple([stat.get(data=lobby_data[col], q=None) for col in SUM_LST])
        return self

    def get_dict(self):
        """Return a dict"""
        return {SUM_LST[ind]: val for ind, val in enumerate(self.team)}, {SUM_LST[ind]: val for ind, val in enumerate(self.lobby)}

    def __repr__(self):
        return 'Game ' + str(self.position)
