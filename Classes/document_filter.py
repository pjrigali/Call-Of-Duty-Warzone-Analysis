from typing import Optional, List
from dataclasses import dataclass
import pandas as pd


@dataclass
class DocumentFilter:

    def __init__(self,
                 original_df: pd.DataFrame,
                 map_choice: Optional[str] = None,
                 mode_choice: Optional[str] = None,
                 username: Optional[str] = None,
                 uno: Optional[int] = None,
                 username_dic: Optional[dict] = None,
                 username_lst: Optional[List[str]] = None,
                 ):

        data = original_df.copy()
        if map_choice:
            data = data[data['map'] == map_choice]

        if mode_choice:
            data = data[data['mode'] == mode_choice]

        if username:
            if username_dic is None:
                raise AttributeError('Need to pass a dict. Example {name or gamertag: uno number, ...}')
            else:
                data = data[data['uno'] == username_dic[username]]

        if uno:
            data = data[data['uno'] == uno]

        if username_lst:
            if username_dic is None:
                raise AttributeError('Need to pass a dict. Example {name or gamertag: uno number, ...}')
            else:
                u_lst = [username_dic[i] for i in username_lst]
                data_lst = list(data['uno'])
                data = data.iloc[[i for i, j in enumerate(data_lst) if str(j) in u_lst]]

        self._df = data.sort_values('startDateTime', ascending=True).reset_index(drop=True)
        self._unique_id_lst = list(self.df['matchID'].unique())
        self._id_lst = list(self.df['matchID'])
        self._map = map_choice
        self._mode = mode_choice
        self._username = username
        self._uno = uno
        self._username_lst = username_lst

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def map_choice(self) -> Optional[str]:
        return self._map

    @property
    def mode_choice(self) -> Optional[str]:
        return self._mode

    @property
    def uno(self) -> Optional[str]:
        return self._uno

    @property
    def username(self) -> Optional[str]:
        return self._username

    @property
    def username_lst(self) -> Optional[List[str]]:
        return self._username_lst

    @property
    def unique_ids(self) -> Optional[List[str]]:
        return self._unique_id_lst

    @property
    def ids(self) -> Optional[List[str]]:
        return self._id_lst
