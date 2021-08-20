from typing import Optional, List
from dataclasses import dataclass
import pandas as pd


@dataclass
class DocumentFilter:

    original_df: pd.DataFrame
    _map: Optional[str]
    _mode: Optional[str]
    _username: Optional[str]
    _uno: Optional[int]
    _username_dict: Optional[dict]
    _username_lst: Optional[List[str]]
    data: pd.DataFrame

    def __init__(self,
                 original_df: pd.DataFrame,
                 _map: str = None,
                 _mode: str = None,
                 _username: str = None,
                 _uno: int = None,
                 _username_dic: dict = None,
                 _username_lst: List[str] = None):
        self.original_df = original_df
        self._map = _map
        self._mode = _mode
        self._username = _username
        self._uno = _uno
        self._username_dic = _username_dic
        self._username_lst = _username_lst

        data = original_df.copy()
        if _map:
            data = data[data['map'] == _map]

        if _mode:
            data = data[data['mode'] == _mode]

        if _username:
            data = data[data['uno'] == _username_dic[_username]]

        if _uno:
            data = data[data['uno'] == _uno]

        if _username_lst:
            u_lst = [_username_dic[i] for i in _username_lst]
            data_lst = list(data['uno'])
            data = data.iloc[[i for i, j in enumerate(data_lst) if str(j) in u_lst]]

        self.data: pd.DataFrame = data.sort_values('startDateTime', ascending=True).reset_index(drop=True)


