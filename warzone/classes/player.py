"""Person class object.

Usage:
 ./warzone/classes/player.py

Author:
 Peter Rigali - 2021-03-29
"""
from dataclasses import dataclass
import pandas as pd
from warzone.classes.document_filter import DocumentFilter
from warzone.utils.class_functions import _get_indexes, _build_time, _build_stats, _build_weapons


@dataclass
class Person:
    """

    The Person class is used to gather all map/mode stats for a given player

    :param original_df: Input data.
    :type original_df: pd.DataFrame
    :param uno: Input person uno Id.
    :type uno: str
    :param gamertag: Input person's gamertag.
    :type gamertag: str
    :example: *None*
    :note: *None

    """

    __slots__ = ["uno", "gamertag", "index_lst", "performance_dict", "favorite_dict", "_original_df"]

    def __init__(self, original_df: pd.DataFrame, uno: str, gamertag: str):
        self.uno = uno
        self.gamertag = gamertag
        self.index_lst = _get_indexes(original_df=original_df, uno=uno)
        self.performance_dict = None
        self.favorite_dict = None
        self._original_df = original_df

    def get_performance(self, map_choice, mode_choice, team_size, stat_type: str = 'sum',
                        stats_weapon_time: str = "stats", position: str = "all", favorite: bool = False) -> dict:
        df = DocumentFilter(input_df=self._original_df.iloc[list(self.index_lst)], map_choice=map_choice,
                            mode_choice=mode_choice, team_size=team_size, position=position).df
        if stats_weapon_time == "stats":
            dic = _build_stats(df=df, stat_type=stat_type)
        elif stats_weapon_time == "weapons":
            dic = _build_weapons(df=df, stat_type=stat_type)
        elif stats_weapon_time == "time":
            dic = _build_time(df=df, stat_type=stat_type)
        else:
            raise AttributeError("stat_type must be one of the following strings (stats, weapons, time)")

        if favorite:
            self.favorite_dict[map_choice][mode_choice][team_size][stats_weapon_time][stat_type] = dic
        else:
            self.performance_dict[map_choice][mode_choice][team_size][stats_weapon_time][stat_type] = dic
        return dic

    def __repr__(self):
        return self.gamertag
