"""Squad class object.

Usage:
 ./squad.py

Author:
 Peter Rigali - 2021-08-30
"""
from typing import Dict, Tuple
from dataclasses import dataclass
import pandas as pd
from warzone.classes.player import Person


@dataclass
class Squad:
    """
    Calculate stats for all maps/modes for each squad memeber.

    :param squad_lst: List of gamertags. Include your gamertag in the list.
    :type squad_lst: Tuple[str]
    :param original_df: Original DataFrame for stats to be calculated from.
    :type original_df: pd.DataFrame
    :param uno_name_dic: A dict of all gamertags and respective unos.
    :type uno_name_dic: dict
    :example: *None*
    :note: This will calculate and return the stats for all squad members.

    """

    __slots__ = ["squad_stats"]

    def __init__(self, squad_lst: Tuple[str], original_df: pd.DataFrame, uno_name_dic: Dict[str, str],
                 build_all: bool = False, favorite: dict = None):
        self.squad_stats = {}
        for teammate in squad_lst:
            p = Person(gamertag=teammate, original_df=original_df, uno=uno_name_dic[teammate])
            if build_all == False and favorite is not None:
                p.favorite_dict = {favorite['fav_map']: {favorite['fav_mode']: {favorite['fav_team_size']: {}}}}
                for stats_weapon_time in ['stats', 'weapons', 'time']:
                    p.favorite_dict[favorite['fav_map']][favorite['fav_mode']][favorite['fav_team_size']][stats_weapon_time] = {}
                    for stat_type in ['sum', 'mean', 'median', 'max']:
                        p.get_performance(map_choice=favorite['fav_map'], mode_choice=favorite['fav_mode'],
                                          team_size=favorite['fav_team_size'], stat_type=stat_type,
                                          stats_weapon_time=stats_weapon_time, position="all", favorite=True)
            elif build_all == True and favorite is None:
                p.performance_dict = {}
                for _map in ['rebirth', 'verdansk', 'caldera']:
                    p.performance_dict[_map] = {}
                    for _mode in ['royale', 'resurgence']:
                        p.performance_dict[_map][_mode] = {}
                        for _team_size in ['solo', 'duo', 'trio', 'quad']:
                            p.performance_dict[_map][_mode][_team_size] = {}
                            for stats_weapon_time in ['stats', 'weapons', 'time']:
                                p.performance_dict[_map][_mode][_team_size][stats_weapon_time] = {}
                                for stat_type in ['sum', 'mean', 'median', 'max']:
                                    p.get_performance(map_choice=_map, mode_choice=_mode, team_size=_team_size,
                                                      stat_type=stat_type, stats_weapon_time=stats_weapon_time,
                                                      position="all", favorite=False)
            self.squad_stats[teammate] = p

    def __repr__(self):
        return 'Squad Stats'
