"""DocumentFilter class object.

Usage:
 ./warzone/classes/document_filter.py

Author:
 Peter Rigali - 2021-08-30
"""
from dataclasses import dataclass
from typing import List, Dict, Union
import pandas as pd
import datetime
from warzone.utils.class_functions import uno_username_dict, apply_filter


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
        >>> from warzone.classes.document_filter import DocumentFilter
        >>> doc = DocumentFilter(input_df=cod.our_df, map_choice='mp_e', mode_choice='quad')
    :note: All inputs, except original_df,  are *Optional* amd defaults are set to None.
        This will return any data with map = rebirth and mode = Quads.
        By specifiying 'cod.our_df', this will only return data related to the user.

    """

    __slots__ = ["df", "unique_match_ids", "match_ids", "map_choice", "mode_choice", "team_size", "username", "uno",
                 "username_dic", "sort_data", "return_empty", "position", "return_sessions", "len"]

    def __init__(self,
                 input_df: pd.DataFrame = None,
                 map_choice: Union[str, List[str]] = None,
                 mode_choice: Union[str, List[str]] = None,
                 team_size: Union[str, List[str]] = None,
                 username: Union[str, List[str]] = None,
                 uno: Union[str, List[str]] = None,
                 username_dic: Dict[str, str] = None,
                 sort_dataframe: str = 'startDateTime',
                 return_empty: bool = True,
                 position: str = 'all',
                 args: dict = None,
                 ):

        if args:
            new = {"input_df": input_df, "map_choice": map_choice, "mode_choice": mode_choice,
                   "team_size": team_size, "username": username, "uno": uno, "username_dic": username_dic,
                   "sort_dataframe": sort_dataframe, "return_empty": return_empty, "position": position,
                   "args": args}

            for key, val in new.items():
                if key not in args.keys():
                    args[key] = new[key]

            input_df = args["input_df"]
            map_choice = args["map_choice"]
            mode_choice = args["mode_choice"]
            team_size = args["team_size"]
            username = args["username"]
            uno = args["uno"]
            username_dic = args["username_dic"]
            sort_dataframe = args["sort_dataframe"]
            return_empty = args["return_empty"]
            position = args["position"]

        data = input_df.copy()
        if map_choice:
            data = apply_filter(data=data, col='map', val=map_choice, dic=None, return_empty=return_empty)

        if mode_choice:
            data = apply_filter(data=data, col='mode', val=mode_choice, dic=None, return_empty=return_empty)

        if team_size:
            data = apply_filter(data=data, col='teamSize', val=team_size, dic=None, return_empty=return_empty)

        if username:
            if username_dic is None:
                username_dic = uno_username_dict(data=data)
            data = apply_filter(data=data, col='uno', val=username, dic=username_dic, return_empty=return_empty)

        if uno:
            data = apply_filter(data=data, col='uno', val=uno, dic=None, return_empty=return_empty)

        if position:
            if position == 'all':
                data = data
            elif position == 'first':
                data = apply_filter(data=data, col='teamPlacement', val=[1], dic=None, return_empty=return_empty)
            else:
                if mode_choice == "royale":
                    val = list(range(1, 10))
                else:
                    val = list(range(1, 5))
                data = apply_filter(data=data, col='teamPlacement', val=val, dic=None, return_empty=return_empty)

        self.df = data.sort_values(sort_dataframe, ascending=True).reset_index(drop=True)
        self.unique_match_ids = tuple(self.df['matchID'].unique().tolist())
        self.match_ids = tuple(self.df['matchID'].tolist())
        self.map_choice = map_choice
        self.mode_choice = mode_choice
        self.team_size = team_size
        self.username = username
        self.uno = uno
        self.username_dic = username_dic
        self.sort_data = sort_dataframe
        self.return_empty = return_empty
        self.position = position
        self.len = self.df.shape[0]

    def get_sessions(self, minutes: int = 60):
        """Splits games into sessions based on a threshold between games"""
        id_dic, ind_dic, count, past_game = {}, {}, 0, None
        for ind, row in self.df.iterrows():
            if past_game is None:
                past_game, id_dic[row['matchID']], ind_dic[count] = row['endDateTime'], True, [ind]
                continue
            elif row['matchID'] in id_dic:
                ind_dic[count].append(ind)
            else:
                if row['startDateTime'] - past_game <= datetime.timedelta(minutes=minutes):
                    past_game, id_dic[row['matchID']] = row['endDateTime'], True
                    ind_dic[count].append(ind)
                else:
                    count += 1
                    past_game, ind_dic[count] = row['endDateTime'], [ind]
        session_df_dic = {key: self.df.iloc[val] for key, val in ind_dic.items()}
        return session_df_dic

    def __repr__(self):
        return 'DocumentFilter'
