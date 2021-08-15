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
from bs4 import BeautifulSoup
import requests
import json

from Utils.gun_dictionary import gun_dict
from Utils.scrape import refresh_data, connect_to_api, clean_api_data
from Utils.creds import user_inputs
from Utils.base import filter_df, running_mean, cum_mean, normalize
from Utils.regress import regression_calcs, regress
from Utils.analysis import get_person_data, get_daily_hourly_weekday_stats
from Utils.analysis import placement_descriptive_stats, first_top5_bottom_stats, bucket, previous_next_placement
from Utils.analysis import weekly_stats, daily_stats, match_difficulty, squad_score_card, get_weapons, find_hackers, meta_weapons
from Utils.plots import personal_plot, lobby_plot, squad_plot

from Utils.medium_posts import deaths_per_circle, engagement_mm, hackers_overtime, find_hackers_from_hacker_df, squad_effect


pd.set_option('display.max_columns', None)


class CallofDuty:
    
    def __init__(self,
                 file: str = 'Personal_Match_Data.csv',
                 refresh: bool = False,
                 hacker_data: bool = False,
                 get_headers: bool = False,
                 ):
        
        self.file: str = file
        self.USERNAME: str = user_inputs['username']
        self.PASSWORD: str = user_inputs['password']
        self.DRIVER_PATH: str = user_inputs['driverpath']
        self.repo: str = user_inputs['repo']
        self.CodTrackerID: str = user_inputs['codtrackerid']
        self.squad: List[str] = user_inputs['squad']
        self.gamertag: str = user_inputs['gamertag']
        self.gun_dic: dict = gun_dict
        self.whole: pd.DataFrame = self._evaluate_df()
        self.last_match_date_time = list(self.whole['startDateTime'])[-1]

        if get_headers:
            self.header = requests.get('https://httpbin.org/headers',
                                       headers=user_inputs['headers']).json()['headers']['User-Agent']
        else:
            self.header = user_inputs['headers']

        if refresh:
            self.whole = refresh_data(old_df=self.whole,
                                      last_match_timestamp=self.last_match_date_time,
                                      repo=self.repo,
                                      filename=None,
                                      savedf=True)
        
        self.match_id_lst = set(self.whole['matchID'])
        self.name_uno_dict = self._get_match_id_set(data=self.whole)
        self.my_uno = self.name_uno_dict[user_inputs['gamertag']]
        self.our_df, self.other_df = self._get_our_and_other_df(data=self.whole)

        if hacker_data:
            self.hacker_df = self._evaluate_df(link='hacker_df.csv')
            self.name_uno_dict_hacker = self._get_match_id_set(data=self.hacker_df)
    
    def _evaluate_df(self, link: str = 'Personal_Match_Data_v4.csv') -> pd.DataFrame:
        df = pd.read_csv(self.repo + link, index_col='Unnamed: 0').drop_duplicates(keep='first')
        start_time_utc_lst = list(df['utcStartSeconds'])
        df['startDateTime'] = [datetime.datetime.utcfromtimestamp(i) for i in start_time_utc_lst]
        end_time_utc_lst = list(df['utcEndSeconds'])
        df['endDateTime'] = [datetime.datetime.utcfromtimestamp(i) for i in end_time_utc_lst]
        day_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        start_time_timestamp_lst = list(df['startDateTime'])
        df['weekDay'] = [day_dic[i.weekday()] for i in start_time_timestamp_lst]
        star_date_time_lst = list(df['startDateTime'])
        df['startDate'] = [datetime.datetime.strftime(i, '%Y-%m-%d') for i in star_date_time_lst]
        df['startTime'] = [datetime.datetime.strftime(i, '%H:%M:%S') for i in star_date_time_lst]
        df['placementPercent'] = (1 - df['teamPlacement'] / df['teamCount']).round(2)

        headshot_lst = list(df['headshots'])
        kill_lst = list(df['kills'])
        ran = range(len(df))

        headshot_ratio_lst = []
        for ind in ran:
            if headshot_lst[ind] == 0 or kill_lst[ind] == 0:
                headshot_ratio_lst.append(0.0)
            else:
                headshot_ratio_lst.append(headshot_lst[ind] / kill_lst[ind])

        # df['headshotRatio'] = df['headshots'] / df['kills']
        df['headshotRatio'] = headshot_ratio_lst
        map_lst = list(df['map'])
        df['map'] = ['mp_e' if 'mp_e' in i else 'mp_d' for i in map_lst]
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

        temp_lst = []
        for val in df['mode']:
            if 'quad' in val:
                temp_lst.append('quad')
            elif 'trio' in val:
                temp_lst.append('trio')
            elif 'duo' in val:
                temp_lst.append('duo')
            elif 'solo' in val:
                temp_lst.append('solo')
            else:
                temp_lst.append('other')
        df['mode'] = temp_lst

        return df.sort_values('startDateTime', ascending=True).reset_index(drop=True)
    
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
        our_df, other_df = data.iloc[list(comb_dic.keys())].copy(), data.iloc[other].copy()

        col_lst = ['headshots', 'kills', 'deaths', 'kdRatio', 'scorePerMinute', 'distanceTraveled',
                   'objectiveBrKioskBuy', 'percentTimeMoving', 'longestStreak', 'damageDone', 'damageTaken',
                   'missionsComplete', 'objectiveLastStandKill', 'objectiveBrDownEnemyCircle1',
                   'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3', 'objectiveBrDownEnemyCircle4',
                   'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6', 'objectiveTeamWiped',
                   'objectiveReviver','headshotRatio', 'objectiveMunitionsBoxTeammateUsed',
                   'objectiveBrCacheOpen', 'objectiveMedalScoreKillSsRadarDrone']

        # Build our Mu's for comparison.
        our_data_dic = {}
        squad_uno_lst = [self.name_uno_dict[i] for i in self.squad]
        for _map in ['mp_e', 'mp_d']:
            our_data_n = our_df[our_df['map'] == _map]
            temp = our_data_n.set_index('uno').loc[squad_uno_lst]
            our_data_dic[_map] = {col: np.mean(temp[col].fillna(0.0)) for col in col_lst}

        col_dic = {'mp_d':
                       {'above': ['deaths', 'objectiveBrKioskBuy', 'missionsComplete',
                                  'objectiveMedalScoreKillSsRadarDrone'],
                        'below': ['headshots', 'kills', 'kdRatio', 'scorePerMinute', 'percentTimeMoving',
                                  'longestStreak', 'damageDone', 'objectiveLastStandKill',
                                  'objectiveBrDownEnemyCircle1',
                                  'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3',
                                  'objectiveBrDownEnemyCircle4', 'objectiveBrDownEnemyCircle5',
                                  'objectiveBrDownEnemyCircle6', 'objectiveTeamWiped', 'headshotRatio',
                                  'objectiveBrCacheOpen']},
                   'mp_e':
                       {'above': ['deaths', 'objectiveBrKioskBuy', 'damageTaken', 'missionsComplete',
                                  'objectiveMunitionsBoxTeammateUsed', 'objectiveBrCacheOpen',
                                  'objectiveMedalScoreKillSsRadarDrone'],
                        'below': ['headshots', 'kills', 'kdRatio', 'scorePerMinute', 'distanceTraveled',
                                  'percentTimeMoving', 'longestStreak', 'damageDone', 'objectiveLastStandKill',
                                  'objectiveBrDownEnemyCircle1', 'objectiveBrDownEnemyCircle2',
                                  'objectiveBrDownEnemyCircle3', 'objectiveBrDownEnemyCircle5',
                                  'objectiveBrDownEnemyCircle6', 'objectiveTeamWiped', 'objectiveReviver',
                                  'headshotRatio']}
                   }

        dic = {i: [] for i in list(other_df.index)}
        for _map in ['mp_e', 'mp_d']:
            t = other_df.iloc[list(np.where(other_df['map'] == _map)[0])]
            for direction in ['above', 'below']:
                for criteria in col_dic[_map][direction]:
                    if direction == 'above':
                        tt = list(t[t[criteria].fillna(0.0) < our_data_dic[_map][criteria]].index)
                    else:
                        tt = list(t[t[criteria].fillna(0.0) > our_data_dic[_map][criteria]].index)
                    tt_dic = {i: True for i in tt}
                    for key in list(t.index):
                        if key in tt_dic:
                            dic[key].append(1)
                        else:
                            dic[key].append(0)

        our_df['hackerProb'] = 0.0
        other_df['hackerProb'] = [np.mean(np.nan_to_num(i))for i in dic.values()]

        return our_df, other_df
    

if __name__ == '__main__':
    start_timen = time.time()
    cod = CallofDuty(refresh=False,
                     hacker_data=False,
                     get_headers=False)
    print(''), print('Cod Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))

    cod
    # data = filter_df(data=cod.our_df, _map='mp_e', _mode='quad', username_dic=cod.name_uno_dict, username='Claim')
    # engagement_info = engagement_mm(data=data)

    # data1 = filter_df(data=cod.other_df, _map='mp_d', _mode='quad')
    # data2 = filter_df(data=cod.other_df, _map='mp_e', _mode='quad')
    # hot1 = hackers_overtime(data=data1)
    # hot2 = hackers_overtime(data=data2)

    squad_data = filter_df(data=cod.our_df, _map='mp_e', _mode='quad')
    squad_effect_info = squad_effect(data=squad_data, username=cod.gamertag, username_dic=cod.name_uno_dict)

    cod
    # start_timen = time.time()
    # person_info = get_person_data(person_lst=cod.squad,
    #                               data=cod.our_df,
    #                               uno_dict=cod.name_uno_dict,
    #                               map_choice='mp_escape')
    # print(''), print('Person_info Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))
    #
    # start_timen = time.time()
    # daily_info, hourly_info, weekday_info = get_daily_hourly_weekday_stats(person='Claim',
    #                                                                        data=cod.our_df,
    #                                                                        save=False)
    # print(''), print('Daily, Hourly, and Weekday_info Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))
    #
    # start_timen = time.time()
    # weapon_info = get_weapons(data=cod.our_df,
    #                           person='Claim',
    #                           uno_dict=cod.name_uno_dict,
    #                           map_choice='mp_e',
    #                           columns=None,
    #                           sort_by=None,
    #                           save=False)
    # print(''), print('Weapon_info Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))
    #
    # start_timen = time.time()
    # placement_info = placement_descriptive_stats(our_data=cod.our_df,
    #                                              other_data=cod.our_df,
    #                                              col='kdRatio',
    #                                              _map='mp_e',
    #                                              _name='Claim',
    #                                              _dic=cod.name_uno_dict,
    #                                              _internal=True)
    # print(''), print('Placement_info Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))
    #
    # start_timen = time.time()
    # winners_info = first_top5_bottom_stats(data=cod.whole,
    #                                        col='kdRatio',
    #                                        _map='mp_e')
    # print(''), print('Winners_info Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))
    #
    # start_timen = time.time()
    # bucket_info = bucket(data=cod.whole,
    #                      placement=[0, 6],
    #                      col_lst=['kdRatio', 'scorePerMinute', 'percentTimeMoving', 'kills', 'deaths'],
    #                      _map='mp_e')
    # bucket_info['ourPlace'] = [np.mean(cod.our_df[cod.our_df['matchID'] == id]['teamPlacement']) for id in
    #                            bucket_info.index]
    # bucket_corr = bucket_info.corr(method='pearson')['ourPlace'].sort_values()
    # print(''), print('Bucket_info Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))
    #
    # start_timen = time.time()
    # previous_next_info = previous_next_placement(data=cod.our_df,
    #                                              _map='mp_e')
    # print(''), print('Previous_next_info Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))
    #
    # start_timen = time.time()
    # weekly_stats = weekly_stats(data=cod.our_df,
    #                             _map='mp_e',
    #                             uno_dic=cod.name_uno_dict,
    #                             name='Claim')
    # print(''), print('Weekly_stats Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))
    #
    # start_timen = time.time()
    # daily_stats = daily_stats(data=cod.our_df,
    #                           _map='mp_e',
    #                           uno_dic=cod.name_uno_dict,
    #                           name='Claim')
    # print(''), print('Daily_stats Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))
    #
    # start_timen = time.time()
    # hardness_info = match_difficulty(our_df=cod.our_df,
    #                                  other_df=cod.other_df,
    #                                  _map='mp_e',
    #                                  test=False,
    #                                  mu_lst=None,
    #                                  sum_lst=None)
    # print(''), print('Match Difficulty Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))
    # ['teamPlacement', 'kdRatio', 'kills', 'deaths', 'objectiveBrDownEnemyCircle1', 'objectiveBrDownEnemyCircle2',
    #  'objectiveBrDownEnemyCircle3', 'objectiveBrDownEnemyCircle4', 'objectiveBrDownEnemyCircle5',
    #  'objectiveBrDownEnemyCircle6', 'headshots', 'assists', 'percentTimeMoving', 'distanceTraveled', 'timePlayed',
    #  'damageDone', 'damageTaken', 'longestStreak', 'scorePerMinute', 'missionsComplete', 'matchID']

    # personal_plot(data=cod.our_df,
    #               username='Claim',
    #               user_dic=cod.name_uno_dict,
    #               col_lst=[],
    #               _map='mp_e')
    #
    # squad_plot(data=cod.our_df,
    #            username='Claim',
    #            username_lst=['Claim', 'MONEYMIKE0410', 'LeoxGemini', 'TheKing109', 'Rhino5378', 'spectator95', 'IAmLordeYahYaYa', 'ninjanapes'],
    #            user_dic=cod.name_uno_dict,
    #            col_lst=['kdRatio', 'kills', 'deaths', 'placementPercent', 'headshots', 'damageDone', 'damageTaken'],
    #            _map='mp_e')
    #
    # lobby_plot(data=cod.other_df,
    #            _map='mp_e')

    # squad_info = squad_score_card(data=cod.our_df,
    #                               usernames=cod.squad,
    #                               username_dic=cod.name_uno_dict,
    #                               _map='mp_e')
    #
    # weapon_info = get_weapons(data=cod.our_df,
    #                           username='Claim',
    #                           username_dic=cod.name_uno_dict,
    #                           _map='mp_e')
    # start_timen = time.time()
    # hacker_info = find_hackers(data=cod.other_df,
    #                            y_column='kills',
    #                            col_lst=['headshots'],
    #                            _map='mp_e')
    # print(''), print('Hacker Info Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))

    # start_timen = time.time()
    # meta_info_base, meta_info_percent = meta_weapons(data=cod.other_df,
    #                                                  _map='mp_e',
    #                                                  top_one=None,
    #                                                  top_five=None)
    # meta_last_14 = pd.DataFrame(meta_info_percent.loc[list(meta_info_percent.index)[-14:]].mean().sort_values(0, ascending=False), columns=['Percent']).round(3)
    # meta_last_14.to_csv('meta_last_14_days.csv')
    # print(''), print('Meta Info Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))

    # hackers = find_hackers_from_hacker_df(data=cod.hacker_df,
    #                                       our_data=cod.our_df,
    #                                       username_dic=cod.name_uno_dict,
    #                                       squad_lst=cod.squad)

    cod

    # {'peppa pig did it', 'tbaginator', 'quit to desktop', 'graha tia', 'va fan culoooo', 'cappin joe', 'timmy',
     # 'sotrash21', 'imagine hacking', 'mclovin', 'number 1 hacker', 'urgirlathot', 'hoff', 'lick my zorch', 'urgirlaho',
     # 'funkymonk', 'cheating is 2 ez', 'hellboy', 'bean shooter', 'nickmercs', 'iluv hearin ucry', 'jgredy8'}
    ['objectiveBrDownEnemyCircle1',
    'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3', 'objectiveBrDownEnemyCircle4',
    'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6']

    temp_whole_rebirth = deaths_per_circle(data=cod.whole, _map='mp_e', _mode='quad')
    temp_hacker_rebirth = deaths_per_circle(data=cod.hacker_df, _map='mp_e', _mode='quad')
    temp_whole_verdansk = deaths_per_circle(data=cod.whole, _map='mp_d', _mode='quad')
    temp_hacker_verdansk = deaths_per_circle(data=cod.hacker_df, _map='mp_d', _mode='quad')

    for i in [temp_hacker_rebirth, temp_whole_rebirth, temp_hacker_verdansk, temp_whole_verdansk]:
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.table(cellText=np.around(i.values, 1),
                 colLabels=i.columns,
                 rowLabels=i.index,
                 loc='center')
        fig.tight_layout()
        plt.show()

    lst = ['objectiveBrDownEnemyCircle1_kill_mean', 'objectiveBrDownEnemyCircle2_kill_mean',
           'objectiveBrDownEnemyCircle3_kill_mean', 'objectiveBrDownEnemyCircle4_kill_mean',
           'objectiveBrDownEnemyCircle5_kill_mean', 'objectiveBrDownEnemyCircle6_kill_mean']

    temp_lst = []
    for temp in [temp_hacker_verdansk, temp_whole_verdansk, temp_hacker_rebirth, temp_whole_rebirth]:
        val_lst = []
        for ind in lst:
            y = temp.loc[ind]['Deaths Per Circle']
            val_lst.append(temp.loc[ind]['Deaths Per Circle'])
        temp_lst.append(val_lst)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.subtract(150, np.cumsum(temp_lst[1])), label='Verdansk', color='tab:blue', alpha=0.75)
    ax.plot(np.subtract(150, np.cumsum(temp_lst[0])), label='Verdansk - Hackers', color='tab:blue', alpha=0.25)
    ax.plot(np.subtract(40, np.cumsum(temp_lst[3])), label='Rebirth', color='tab:orange', alpha=0.75)
    ax.plot(np.subtract(40, np.cumsum(temp_lst[2])), label='Rebirth - Hackers', color='tab:orange', alpha=0.25)
    ax.set_ylabel('Deaths')
    ax.set_xlabel('Circle')
    ax.set_title('Matches vs Matches with Hackers', fontsize='xx-large')
    ax.grid(linewidth=1, linestyle=(0, (5, 5)), alpha=.75)
    ax.legend(loc='upper right', fontsize='large', frameon=True, framealpha=0.85)
    plt.show()

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

