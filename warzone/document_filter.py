"""DocumentFilter class object.

Usage:
 ./document_filter.py

Author:
 Peter Rigali - 2021-08-30
"""
from typing import Optional, List, Dict, Union
from dataclasses import dataclass
import pandas as pd


def _check_empty(data: pd.DataFrame, return_empty: bool = True) -> pd.DataFrame:
    """Checks if the input dataframe is empty"""
    if return_empty:
        return data
    else:
        if data.empty:
            raise AttributeError('Based on input params, the dataframe will be empty.')
        else:
            return data


def get_uno_username_dict(data: pd.DataFrame) -> dict:
    """Return a dict {gamertag: uno, gamertag1: uno1, etc}"""
    if 'uno' not in data.columns:
        raise AttributeError('uno required in dataframe')
    if 'username' not in data.columns:
        raise AttributeError('username required in dataframe')

    comb_set = (data['uno'] + '-splitpoint-' + data['username']).unique().tolist()
    return {i.split('-splitpoint-')[1]: i.split('-splitpoint-')[0] for i in comb_set if type(i) == str}


def _accept_list(data: pd.DataFrame, col: str, lst: List[str], return_empty: bool = True) -> pd.DataFrame:
    """Handles a list of str's as an input"""
    if col not in data.columns:
        raise AttributeError(col + ' not included in the passed dataframe column list')
    else:
        lst_dic = {i: True for i in lst}
        data_lst = data[col].tolist()
        new_data = data.iloc[[i for i, j in enumerate(data_lst) if str(j) in lst_dic]]
        return _check_empty(data=new_data, return_empty=return_empty)


def _accept_str(data: pd.DataFrame, col: str, string: str, return_empty: bool = True) -> pd.DataFrame:
    """Handles a str input"""
    if col not in data.columns:
        raise AttributeError(col + ' not included in the passed dataframe column list')
    else:
        return _check_empty(data=data[data[col] == string], return_empty=return_empty)


def _evaluate_data(data: pd.DataFrame, col: str, val: Union[str, List[str]],
                   dic: Optional[Dict[str, str]] = None, return_empty: bool = True) -> pd.DataFrame:
    """Filters data based on input col and val"""
    if dic is None:
        if type(val) == str:
            return _accept_str(data=data, col=col, string=val, return_empty=return_empty)
        elif type(val) == list:
            return _accept_list(data=data, col=col, lst=val, return_empty=return_empty)
    else:
        if type(val) == str:
            return _accept_str(data=data, col=col, string=dic[val], return_empty=return_empty)
        elif type(val) == list:
            return _accept_list(data=data, col=col, lst=[dic[i] for i in val], return_empty=return_empty)


@dataclass
class DocumentFilter:
    """

    Get a selection from a DataFrame.
    Uses a set of filters to return a desired set of data to be used in later analysis.
    Can handle single string or list of string inputs.

    :param input_df: Input DataFrame to be filtered.
    :type input_df: pd.DataFrame
    :param map_choice: Map filter. Either 'rebirth', 'verdansk', 'caldera', or 'other'. *Optional*
    :type map_choice: str or list
    :param mode_choice: Mode filter. Either 'royale', 'resurgence', 'plunder', or 'other'. *Optional*
    :type mode_choice: str or list
    :param team_size: Team Size filter. Either 'solo', 'duo', 'trio', or 'quad'. *Optional*
    :type team_size: str or list
    :param username: Filter by a players username. Can cause errors if same username as another player. *Optional*
    :type username: str or list
    :param uno: Filter by a players uno. *Optional*
    :type uno: str or list
    :param username_dic: If 'username' is used, will create if not passed. {username1: uno1, etc}. *Optional*
    :type username_dic: dict
    :param sort_dataframe: If passed will sort dataframe using this column.
    :type sort_dataframe: str
    :param return_empty: If False, will not return empty DataFrame.
    :type return_empty: bool
    :example:
        >>> from warzone.document_filter import DocumentFilter
        >>> doc = DocumentFilter(input_df=cod.our_df, map_choice='mp_e', mode_choice='quad')
    :note: All inputs, except original_df,  are *Optional* amd defaults are set to None.
        This will return any data with map = rebirth and mode = Quads.
        By specifiying 'cod.our_df', this will only return data related to the user.

    """
    def __init__(self,
                 input_df: pd.DataFrame,
                 map_choice: Union[str, List[str]] = None,
                 mode_choice: Union[str, List[str]] = None,
                 team_size: Union[str, List[str]] = None,
                 username: Union[str, List[str]] = None,
                 uno: Union[str, List[str]] = None,
                 username_dic: Optional[Dict[str, str]] = None,
                 sort_dataframe: str = 'startDateTime',
                 return_empty: bool = True,
                 ):

        data = input_df.copy()
        if map_choice:
            data = _evaluate_data(data=data, col='map', val=map_choice, dic=None, return_empty=return_empty)

        if mode_choice:
            data = _evaluate_data(data=data, col='mode', val=mode_choice, dic=None, return_empty=return_empty)

        if team_size:
            data = _evaluate_data(data=data, col='teamSize', val=team_size, dic=None, return_empty=return_empty)

        if username:
            if username_dic is None:
                username_dic = get_uno_username_dict(data=data)
            data = _evaluate_data(data=data, col='uno', val=username, dic=username_dic, return_empty=return_empty)

        if uno:
            data = _evaluate_data(data=data, col='uno', val=uno, dic=None, return_empty=return_empty)

        self._df = data.sort_values(sort_dataframe, ascending=True).reset_index(drop=True)
        self._unique_id_lst = tuple(self.df['matchID'].unique().tolist())
        self._id_lst = tuple(self.df['matchID'].tolist())
        self._map = map_choice
        self._mode = mode_choice
        self._team_size = team_size
        self._username = username
        self._uno = uno
        self._username_dic = username_dic
        self._sort_data = sort_dataframe
        self._return_empty = return_empty
        self._len = self._df.shape[0]

    def __getitem__(self):
        return self._df

    def __repr__(self):
        return 'DocumentFilter'

    @property
    def len(self) -> int:
        """Returns length of output Dataframe"""
        return self._len

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
        """Returns the map_choice used to filter"""
        return self._map

    @property
    def mode_choice(self) -> Optional[str]:
        """Returns the mode_choice used to filter"""
        return self._mode

    @property
    def team_size(self) -> Optional[str]:
        """Returns the team_size used to filter"""
        return self._team_size

    @property
    def uno(self) -> Optional[str]:
        """Returns the Uno used to filter"""
        return self._uno

    @property
    def username(self) -> Optional[str]:
        """Returns the Username used to filter"""
        return self._username

    @property
    def unique_match_ids(self) -> tuple:
        """Returns unique match ids from the filtered DataFrame"""
        return self._unique_id_lst

    @property
    def match_ids(self) -> tuple:
        """Returns match ids from the filtered DataFrame"""
        return self._id_lst

    @property
    def username_dic(self) -> Optional[Dict[str, str]]:
        """Returns {username: uno} dict"""
        return self._username_dic

    @property
    def sort_criteria(self) -> str:
        """Returns column used for sorting"""
        return self._sort_data

    @property
    def return_empty(self) -> bool:
        """Returns if the class will accept an empty result"""
        return self._return_empty
