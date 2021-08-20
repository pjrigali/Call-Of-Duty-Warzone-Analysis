from typing import List, Optional
import pandas as pd
import numpy as np
from Classes.document_filter import DocumentFilter


class Person:

    # Whole data per gamer
    gamertag: str = None
    uno: str = None
    total_kills: int = None
    total_deaths: int = None
    total_kd: float = None
    total_headshot_ratio: float = None
    total_games_played: int = None
    total_play_time: int = None
    match_ids: List[str] = None
    whole_index: List[int] = None
    match_rows: pd.DataFrame = None

    # Data for Rebirth
    rebirth_wins: int = None
    rebirth_top_5s: int = None
    rebirth_win_percent: float = None
    rebirth_top_5_percent: float = None
    rebirth_game_count: int = None
    rebirth_average_placement: float = None
    rebirth_time_played: float = None
    rebirth_kills: int = None
    rebirth_deaths: int = None
    rebirth_kd: float = None
    rebirth_average_damage_done: float = None
    rebirth_average_damage_taken: float = None
    rebirth_match_ids: List[str] = None

    # Data for Verdansk
    verdansk_wins: int = None
    verdansk_top_10s: int = None
    verdansk_win_percent: float = None
    verdansk_top_10_percent: float = None
    verdansk_game_count: int = None
    verdansk_average_placement: float = None
    verdansk_time_played: float = None
    verdansk_kills: int = None
    verdansk_deaths: int = None
    verdansk_kd: float = None
    verdansk_average_damage_done: float = None
    verdansk_average_damage_taken: float = None
    verdansk_match_ids: List[str] = None

    kd_ratios: pd.DataFrame = None
    weapons: pd.DataFrame = None


class Squad:

    # squad_lst: List[str] = None
    # df: pd.DataFrame = None
    # uno_name_dic: dict = None

    def __init__(self, squad_lst: List[str], df: pd.DataFrame, uno_name_dic: dict):
        # self.squad_lst = squad_lst
        # self.df = df
        # self.uno_name_dic = uno_name_dic

        squad_dic = {}
        for teammate in squad_lst:
            temp_df = DocumentFilter(original_df=df, _uno=uno_name_dic[teammate]).get_df()
            person = Person()
            person.gamertag = teammate
            person.uno = uno_name_dic[teammate]
            person.total_kills = np.sum(temp_df['kills'])
            person.total_deaths = np.sum(temp_df['deaths'])
            person.total_kd = np.mean(temp_df['kdRatio'])
            person.total_headshot_ratio = np.mean(temp_df['headshotRatio'])
            person.total_games_played = len(temp_df)
            person.total_play_time = np.sum(temp_df['timePlayed'])
            person.match_ids = list(temp_df['matchId'].unique())
            person.whole_index = list(temp_df.index)

            for _map in ['mp_d', 'mp_e']:
                temp_df = DocumentFilter(original_df=df, _map=_map, _uno=person.uno).get_df()

                if _map == 'mp_d':
                    person.verdansk_kills = np.sum(temp_df['kills'])
                    person.verdansk_deaths = np.sum(temp_df['deaths'])
                    person.verdansk_kd = np.mean(temp_df['kdRatio'])
                    person.verdansk_average_damage_done = np.mean(temp_df['damageDone'])
                    person.verdansk_average_damage_taken = np.mean(temp_df['damageTaken'])
                    person.verdansk_game_count = len(temp_df)
                    person.verdansk_time_played = np.sum(temp_df['timePlayed'])
                    person.verdansk_top_10s = len(temp_df[(temp_df['teamPlacement'] <= 10) & (temp_df['teamPlacement'] > 1)])
                    person.verdansk_wins = len(temp_df[temp_df['teamPlacement'] == 1])
                    person.verdansk_average_placement = np.mean(temp_df['placementPercent'])
                    person.verdansk_match_ids = list(temp_df['matchID'].unique())

                    person.verdansk_win_percent = 0.0
                    if person.verdansk_wins != 0:
                        person.verdansk_win_percent = person.verdansk_wins / person.verdansk_game_count

                    person.verdansk_top_10_percent = 0.0
                    if person.verdansk_top_10s != 0:
                        person.verdansk_top_10_percent = person.verdansk_top_10s / person.verdansk_game_count

                else:
                    person.rebirth_kills = np.sum(temp_df['kills'])
                    person.rebirth_deaths = np.sum(temp_df['deaths'])
                    person.rebirth_kd = np.mean(temp_df['kdRatio'])
                    person.rebirth_average_damage_done = np.mean(temp_df['damageDone'])
                    person.rebirth_average_damage_taken = np.mean(temp_df['damageTaken'])
                    person.rebirth_game_count = len(temp_df)
                    person.rebirth_time_played = np.sum(temp_df['timePlayed'])
                    person.rebirth_top_5s = len(temp_df[(temp_df['teamPlacement'] <= 5) & (temp_df['teamPlacement'] > 1)])
                    person.rebirth_wins = len(temp_df[temp_df['teamPlacement'] == 1])
                    person.rebirth_average_placement = np.mean(temp_df['placementPercent'])
                    person.rebirth_match_ids = list(temp_df['matchID'].unique())

                    person.rebirth_win_percent = 0.0
                    if person.rebirth_wins != 0:
                        person.rebirth_win_percent = person.rebirth_wins / person.rebirth_game_count

                    person.rebirth_top_5_percent = 0.0
                    if person.rebirth_top_5s != 0:
                        person.rebirth_top_5_percent = person.rebirth_top_5s / person.rebirth_game_count

            squad_dic[teammate] = person

        self.squad_dic = squad_dic


