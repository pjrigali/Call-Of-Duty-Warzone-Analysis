from dataclasses import dataclass
from typing import Union
import pandas as pd


@dataclass
class Window:

    __slots__ = ('position', 'match_ids', 'team_df', 'lobby_df', 'len', 'games')

    def __init__(self, window: int, match_ids: Union[list, tuple], team: pd.DataFrame, lobby: pd.DataFrame):
        self.position = window
        self.match_ids = match_ids
        self.team_df = team
        self.lobby_df = lobby
        self.len = match_ids.__len__()
        self.games = None

    def __repr__(self):
        return 'Window ' + str(self.position)
