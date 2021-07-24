# -*- coding: utf-8 -*-
"""
Created on Sat May 15 23:18:06 2021

@author: Peter
"""
import pandas as pd
import numpy as np
import datetime
import time
from typing import List
import matplotlib.pyplot as plt

from Utils.get_data import get_person_data, get_daily_hourly_weekday_stats, get_weapons, get_lobby_difficulty
from Utils.gun_dictionary import gun_dict
from Utils.scrape import refresh_data
from Utils.creds import user_inputs
from Utils.outlier import stack, outlierStd, outlierCD, outlierDev, outlierDistance, outlierHist, outlierKNN, outlierRegression
from Utils.regress import regression_calcs, regress
from Utils.analysis import placement_descriptive_stats, first_top5_bottom_stats, bucket, previous_next_placement
from Utils.analysis import weekly_stats, daily_stats, match_difficulty, squad_score_card
from Utils.plots import personal_plot, lobby_plot, squad_plot
pd.set_option('display.max_columns', None)


class CallofDuty:
    
    def __init__(self,
                 file: str = 'Personal_Match_Data.csv',
                 refresh: bool = False,
                 ):
        
        self.file: str = file
        self.USERNAME: str = user_inputs['username']
        self.PASSWORD: str = user_inputs['password']
        self.DRIVER_PATH: str = user_inputs['driverpath']
        self.repo: str = user_inputs['repo']
        self.CodTrackerID: str = user_inputs['codtrackerid']
        self.squad: List[str] = user_inputs['squad']
        self.gun_dic: dict = gun_dict
        self.whole: pd.DataFrame = self._evaluate_df()
        self.last_match_date_time = list(self.whole['startDateTime'])[-1]
            
        if refresh:
            self.whole = refresh_data(old_df=self.whole,
                                      last_match_timestamp=self.last_match_date_time,
                                      repo=self.repo,
                                      filename=None,
                                      savedf=True)
        
        self.match_id_lst = set(self.whole['matchID'])
        self.name_uno_dict = self._get_match_id_set(self.whole)
        self.my_uno = self.name_uno_dict[user_inputs['gamertag']]
        self.our_df, self.other_df = self._get_our_and_other_df(self.whole)
    
    def _evaluate_df(self, link: str = 'Personal_Match_Data_v3.csv') -> pd.DataFrame:
        df = pd.read_csv(self.repo + link, index_col='Unnamed: 0')
        start_time_lst = list(df['utcStartSeconds'])
        df['startDateTime'] = [datetime.datetime.utcfromtimestamp(i) for i in start_time_lst]
        end_time_lst = list(df['utcEndSeconds'])
        df['endDateTime'] = [datetime.datetime.utcfromtimestamp(i) for i in end_time_lst]
        day_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        start_time_timestamp_lst = list(df['startDateTime'])
        df['weekDay'] = [day_dic[i.weekday()] for i in start_time_timestamp_lst]
        df['startDate'] = [datetime.datetime.strftime(i, '%Y-%m-%d') for i in df['startDateTime']]
        df['startTime'] = [datetime.datetime.strftime(i, '%H:%M:%S') for i in df['startDateTime']]
        df['placementPercent'] = (1 - df['teamPlacement'] / df['teamCount']).round(2)
        # Fix Blown out Damage Taken
        large_damage_taken = list(df[df['damageTaken'] > 100000].index)
        for i in large_damage_taken:
            df.loc[i, 'damageTaken'] = df.loc[i, 'damageDone']
        # Convert to Strings, not sure if necessary
        weapon_col_lst = [i for i in df.columns if ('primaryWeapon_' in i) | ('secondaryWeapon_' in i)]
        cols_lst = ['map', 'mode', 'team', 'username', 'uno', 'matchID'] + weapon_col_lst
        for col in cols_lst:
            temp_col_lst = list(df[col])
            df[col] = [str(i) for i in temp_col_lst]
        
        return df
    
    def _get_match_id_set(self, data: pd.DataFrame) -> dict:
        comb_set = set(data['uno'] + '-splitpoint-' + data['username'])
        return {i.split('-splitpoint-')[1]: i.split('-splitpoint-')[0] for i in comb_set}
        
    def _get_our_and_other_df(self, data: pd.DataFrame, name: str = None):
        if name:
            uno = self.name_uno_dict[name]
        else:
            uno = self.my_uno
            
        base_lst = data['matchID'] + '-splitpoint-' + data['team']
        base_our_lst = data[data['uno'] == uno]['matchID'] + '-splitpoint-' + data[data['uno'] == uno]['team']
        our_lst = {i: True for i in base_our_lst}
        comb_dic = {i: True for i, j in enumerate(base_lst) if j in our_lst}
        other = [i for i in data.index if i not in comb_dic]
        return data.iloc[list(comb_dic.keys())], data.iloc[other]
        
    def findHackers(self,
                    y,
                    df,
                    lst
                    ):
        
        if type(y) == str:
            y_n = np.array(df[y])
        elif type(y) == list:
            y_n = np.array(y)
        else:
            print('For Y, provide a string or list')
        
        ind = []
        for i in lst:
            x_n = np.array(df[i])
            x_y = self.stack(x_n, y_n, False)
            analysis = [list(self.outlierDev(x_n, .95)[1][0]),
                        list(self.outlierStd(x_n, 3)[1][1]),
                        list(self.outlierDistance(x_y)),
                        list(self.outlierHist(x_n, 0.75)),
                        list(self.outlierCD(x_n, y_n)[0])]
            ind.append(sum(analysis, []))
        
        temp_dict = {i: 0 for i in set(sum(ind, []))}
        for i in sum(ind, []):
            temp_dict[i] += 1
        
        pot = []
        for i in temp_dict.keys():
            if temp_dict[i] >= 3 * len(lst):
                pot.append(i)
        
        return pot, df.iloc[pot]
    

if __name__ == '__main__':
    start_timen = time.time()
    cod = CallofDuty(refresh=False)
    print(''), print('Cod Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))
    
    start_timen = time.time()
    people = ['Claim', 'MONEYMIKE0410', 'LeoxGemini', 'TheKing109', 'Rhino5378', 'spectator95', 'IAmLordeYahYaYa']
    person_info = get_person_data(person_lst=people,
                                  data=cod.our_df,
                                  uno_dict=cod.name_uno_dict,
                                  map_choice='mp_escape'
                                  )
    print(''), print('Person_info Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))

    start_timen = time.time()
    daily_info, hourly_info, weekday_info = get_daily_hourly_weekday_stats(person='Claim',
                                                                           data=cod.our_df,
                                                                           save=False)
    print(''), print('Daily, Hourly, and Weekday_info Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))

    start_timen = time.time()
    weapon_info = get_weapons(data=cod.our_df,
                              person='Claim',
                              uno_dict=cod.name_uno_dict,
                              map_choice='mp_e',
                              columns=None,
                              sort_by=None,
                              save=False)
    print(''), print('Weapon_info Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))

    start_timen = time.time()
    lobby_info = get_lobby_difficulty(data=cod.other_df,
                                      eval_criteria=None)
    print(''), print('Lobby_info Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))

    start_timen = time.time()
    placement_info = placement_descriptive_stats(our_data=cod.our_df,
                                                 other_data=cod.our_df,
                                                 col='kdRatio',
                                                 _map='mp_e',
                                                 _name='Claim',
                                                 _dic=cod.name_uno_dict,
                                                 _internal=True)
    print(''), print('Placement_info Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))

    start_timen = time.time()
    winners_info = first_top5_bottom_stats(data=cod.whole,
                                           col='kdRatio',
                                           _map='mp_e')
    print(''), print('Winners_info Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))

    start_timen = time.time()
    bucket_info = bucket(data=cod.whole,
                         placement=[0, 6],
                         col_lst=['kdRatio', 'scorePerMinute', 'percentTimeMoving', 'kills', 'deaths'],
                         _map='mp_e')
    bucket_info['ourPlace'] = [np.mean(cod.our_df[cod.our_df['matchID'] == id]['teamPlacement']) for id in
                               bucket_info.index]
    bucket_corr = bucket_info.corr(method='pearson')['ourPlace'].sort_values()
    print(''), print('Bucket_info Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))

    start_timen = time.time()
    previous_next_info = previous_next_placement(data=cod.our_df,
                                                 _map='mp_e')
    print(''), print('Previous_next_info Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))

    start_timen = time.time()
    weekly_stats = weekly_stats(data=cod.our_df,
                                _map='mp_e',
                                uno_dic=cod.name_uno_dict,
                                name='Claim')
    print(''), print('Weekly_stats Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))

    start_timen = time.time()
    daily_stats = daily_stats(data=cod.our_df,
                              _map='mp_e',
                              uno_dic=cod.name_uno_dict,
                              name='Claim')
    print(''), print('Daily_stats Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))

    start_timen = time.time()
    hardness_info = match_difficulty(our_df=cod.our_df,
                                     other_df=cod.other_df,
                                     _map='mp_e',
                                     test=False,
                                     mu_lst=None,
                                     sum_lst=None)
    print(''), print('Match Difficulty Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))
    # ['teamPlacement', 'kdRatio', 'kills', 'deaths', 'objectiveBrDownEnemyCircle1', 'objectiveBrDownEnemyCircle2',
    #  'objectiveBrDownEnemyCircle3', 'objectiveBrDownEnemyCircle4', 'objectiveBrDownEnemyCircle5',
    #  'objectiveBrDownEnemyCircle6', 'headshots', 'assists', 'percentTimeMoving', 'distanceTraveled', 'timePlayed',
    #  'damageDone', 'damageTaken', 'longestStreak', 'scorePerMinute', 'missionsComplete', 'matchID']

    personal_plot(data=cod.our_df,
                  username='Claim',
                  user_dic=cod.name_uno_dict,
                  col_lst=[],
                  _map='mp_e')

    squad_plot(data=cod.our_df,
               username='Claim',
               username_lst=['Claim', 'MONEYMIKE0410', 'LeoxGemini', 'TheKing109', 'Rhino5378', 'spectator95', 'IAmLordeYahYaYa', 'ninjanapes'],
               user_dic=cod.name_uno_dict,
               col_lst=['kdRatio', 'kills', 'deaths', 'placementPercent', 'headshots', 'damageDone', 'damageTaken'],
               _map='mp_e')

    lobby_plot(data=cod.other_df,
               _map='mp_e')

    people = ['Claim', 'MONEYMIKE0410', 'LeoxGemini', 'TheKing109', 'Rhino5378', 'spectator95', 'IAmLordeYahYaYa',
              'ninjanapes']
    squad_info = squad_score_card(data=cod.our_df, usernames=people, username_dic=cod.name_uno_dict, _map='mp_e')

    cod
    
    # w = cod.weapon_results(name='Claim',
    #                        map_specfic=None,
    #                        sort_by='kd',
    #                        limit=10,
    #                        threshold=['kills', 50],
    #                        specfic_weapon=['MP5', 'M4A1'],
    #                        compute_cumsum=True
    #                        )
    #
    #
    # from sklearn.model_selection import train_test_split
    #
    # train, test = train_test_split(np.array(data), test_size=.80)
    #
    # d = cod.match_difficulty(data, plotn=False)
    # matches = list(data['matchID'])
    # data['difficulty'] = [d[i] for i in matches]
    # x = np.array(data['kdRatio'])
    # y = np.array(data['difficulty'])
    # ind, df = cod.findHackers('difficulty', data, ['scorePerMinute', 'kdRatio', 'headshots'])
    #
    # cod.regress(xtrain=x, ytrain=y, xtest=False, ytest=False, test=False, plotn=True)
    # from statsmodels import regression
    #
    # model = regression.linear_model.OLS(y, add_constant(x)).fit()
    # model.summary()
    #
    # cod.getBestWeapons(data, 0.00, 50, 'Claim', False, True)

