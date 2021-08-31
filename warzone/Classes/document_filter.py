"""DocumentFilter class object.

Usage:
 ./document_filter.py

Author:
 Peter Rigali - 2021-08-30
"""
from typing import Optional, List
from dataclasses import dataclass
import pandas as pd


@dataclass
class DocumentFilter:
    """

    Get a selection from a DataFrame.
    Uses a set of filters to return a desired set of data to be used in later analysis.

    :param original_df: Input DataFrame to be filtered.
    :type original_df: pd.DataFrame
    :param map_choice: Map filter. Either 'mp_e' for Rebirth and 'mp_d' for Verdansk. *Optional*
    :type map_choice: str
    :param mode_choice: Mode filter. Either 'solo', 'duo', 'trio', or 'quad'. *Optional*
    :type mode_choice: str
    :param username: Filter by a players username. Can cause errors if same username as another player. *Optional*
    :type username: str
    :param uno: Filter by a players uno. *Optional*
    :type uno: str
    :param username_dic: Required if 'username' or 'username_lst' is used. {username1: uno1, username2: uno2, etc}. *Optional*
    :type username_dic: dict
    :param username_lst: Filter using a list of usernames. *Optional*
    :type username_lst: List[str]
    :example:
        >>> from document_filter import DocumentFilter
        >>> doc = DocumentFilter(original_df=cod.our_df, map_choice='mp_e', mode_choice='quad')
    :note: All inputs, except original_df,  are *Optional* amd defaults are set to None.
        This will return any data with map = rebirth and mode = Quads.
        By specifiying 'cod.our_df', this will only return data related to the user.

    """
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
        self._username_dic = username_dic

    def __repr__(self):
        return 'DocumentFilter'

    @property
    def df(self) -> pd.DataFrame:
        """Returns the filtered DataFrame"""
        return self._df

    @df.setter
    def df(self, val: pd.DataFrame):
        """Set df value"""
        self._df = val

    @property
    def map_choice(self) -> Optional[str]:
        """Returns the map used to filter"""
        return self._map

    @property
    def mode_choice(self) -> Optional[str]:
        """Returns the mode used to filter"""
        return self._mode

    @property
    def uno(self) -> Optional[str]:
        """Returns the uno used to filter"""
        return self._uno

    @property
    def username(self) -> Optional[str]:
        """Returns the username used to filter"""
        return self._username

    @property
    def username_lst(self) -> Optional[List[str]]:
        """Returns the username list used to filter"""
        return self._username_lst

    @property
    def unique_ids(self) -> Optional[List[str]]:
        """Returns unique match ids from the filtered DataFrame"""
        return self._unique_id_lst

    @property
    def ids(self) -> Optional[List[str]]:
        """Returns match ids from the filtered DataFrame"""
        return self._id_lst

    @property
    def username_dic(self) -> Optional[dict]:
        """Returns username: uno dict"""
        return self._username_dic
