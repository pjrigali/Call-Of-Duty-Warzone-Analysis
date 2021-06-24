# -*- coding: utf-8 -*-
"""
Created on Sat May 15 23:18:06 2021

@author: Peter
"""
import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt

from Utils.get_data import get_person_data, get_daily_hourly_weekday_stats, get_weapons, get_lobby_difficulty
from Utils.gun_dictionary import gun_dict
from Utils.scrape import refresh_data
from Utils.creds import user_inputs
from Utils.outlier import stack, outlierStd, outlierCD, outlierDev, outlierDistance, outlierHist, outlierKNN, outlierRegression
from Utils.regress import regression_calcs, regress
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
    
    def _evaluate_df(self, link: str = 'Personal_Match_Data_v2.csv') -> pd.DataFrame:
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
    
    @staticmethod
    def normalize(arr: np.ndarray,
                  multi: bool = False,
                  ):
        
        if multi:
            return np.array([(arr[:, i] - np.min(arr[:, i])) / (np.max(arr[:, i]) - np.min(arr[:, i])) for i in
                             range(arr.shape[1])]).T
        else:
            return np.around((arr - np.min(arr)) / (np.max(arr) - np.min(arr)).T, 3)
    
    def match_difficulty(self,
                         df,
                         plotn: bool = False
                         ):
        
        ind = set(df['matchID'])
        match_indexes = {match: list(df[df['matchID'] == match].index) for match in ind}
        
        def calcs(df, tc):
            one = df['kills'].mean()
            two = df['damageDone'].mean()
            three = df['headshots'].mean()
            four = df['missionsComplete'].sum() / tc
            five = df['scorePerMinute'].mean()
            six = df['kdRatio'].mean()
            return [round(i, 3) for i in [one, two, three, four, five, six]]
        
        result = {match: [] for match in ind}
        for match in ind:
            temp_df = df.iloc[match_indexes[match]]
            team_count = list(temp_df['teamCount'])[0]
            difficulty = []
            for spot, rank in enumerate([.25, .5, 1.0]):
                threshold = np.ceil(team_count * rank)
                if spot == 0:
                    difficulty.append(calcs(temp_df[temp_df['teamPlacement'] <= threshold], team_count))
                    continue
                elif spot == 1:
                    difficulty.append(calcs(temp_df[temp_df['teamPlacement'] <= threshold], team_count))
                    continue
                else:
                    difficulty.append(calcs(temp_df, team_count))
                    continue
            result[match] = difficulty
        
        final = {match: [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] for match in ind}
        for rank in [0, 1, 2]:
            for criteria in [0, 1, 2, 3, 4, 5]:
                temp_lst_n = self.normalize(np.array([result[match][rank][criteria] for match in ind]), False)
                for stat, match in enumerate(ind):
                    final[match][rank][criteria] = temp_lst_n[stat]
        
        hardness = {match: np.around(np.mean(final[match]), 3) for match in ind}
        
        if plotn:
            plt.title('Normailized Match Difficulty')
            plt.plot(range(len(ind)), list(hardness.values()))
            plt.xticks(range(len(ind)), [str(i) for i in ind], rotation=-90)
            plt.show()
        
        return hardness
    

if __name__ == '__main__':
    start_timen = time.time()
    cod = CallofDuty(refresh=False)
    print(''), print('Cod Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))
    
    start_timen = time.time()
    people = ['Claim', 'MONEYMIKE0410', 'LeoxGemini', 'TheKing109', 'Rhino5378', 'spectator95']
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

