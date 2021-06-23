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
from scipy import stats
from statsmodels import regression
from statsmodels.tools import add_constant
from ast import literal_eval
from Utils.scrape import refresh_data
from Utils.outlier import stack, outlierStd, outlierCD, outlierDev, outlierDistance, outlierHist, outlierKNN, outlierRegression
pd.set_option('display.max_columns', None)


class CallofDuty:
    
    def __init__(self,
                 file: str = 'Personal_Match_Data.csv',
                 refresh: bool = False,
                 ):
        
        self.file: str = file
        self.USERNAME: str = 'peterjrigali@gmail.com'
        self.PASSWORD: str = 'Lacrosse89!!'
        self.DRIVER_PATH: str = '/Users/Peter/Desktop/chromedriver'
        self.repo: str = 'C:\\Users\\Peter\\Desktop\\Personal\\11_Repository\\Call of Duty Related\\Data\\Personal Data\\'
        self.CodTrackerID: str = 'Prigali#1499'
        self.gun_dic: dict = {
            # Modern Warfare
            'iw8_ar_tango21': 'RAM-7', 'iw8_ar_mike4': 'M4A1',
            'iw8_ar_valpha': 'AS VAL', 'iw8_ar_falpha': 'FR 5.56',
            'iw8_ar_mcharlie': 'M13', 'iw8_ar_akilo47': 'AK-47',
            'iw8_ar_asierra12': 'Oden', 'iw8_ar_galima': 'AMAX',
            'iw8_ar_sierra552': 'Grau', 'iw8_ar_falima': 'FAL',
            'iw8_ar_anovember94': 'AN', 'iw8_ar_kilo433': 'Kilo',
            'iw8_ar_scharlie': 'FN Scar 17', 'iw8_sh_mike26': 'VLK',
            'iw8_sh_charlie725': '725', 'iw8_sh_oscar12': 'Origin 12',
            'iw8_sh_aalpha12': 'Jack 12', 'iw8_sh_romeo870': 'Model 680',
            'iw8_sh_dpapa12': 'R9', 'iw8_sn_sbeta': 'Mk2 Carbine',
            'iw8_sn_crossbow': 'Crossbow', 'iw8_sn_romeo700': 'SPR',
            'iw8_sn_kilo98': 'Kar', 'iw8_sn_mike14': 'EBR',
            'iw8_sn_sksierra': 'SKS', 'iw8_sn_alpha50': 'AX-50',
            'iw8_sn_hdromeo': 'HDR', 'iw8_sn_delta': 'Dragunov',
            'iw8_sn_xmike109': 'Rytec AMR', 'iw8_lm_kilo121': 'M91',
            'iw8_lm_mkilo3': 'Bruen', 'iw8_lm_mgolf34': 'MG34',
            'iw8_lm_lima86': 'SA87', 'iw8_lm_pkilo': 'PKM',
            'iw8_lm_sierrax': 'FiNN', 'iw8_lm_mgolf36': 'Holger',
            'iw8_la_gromeo': 'PILA', 'iw8_la_rpapa7': 'RPG-7 - mw',
            'iw8_la_juliet': 'JOKR', 'iw8_la_kgolf': 'Strela',
            'iw8_la_mike32': 'Grenade Launcher', 'iw8_pi_cpapa': 'Magnum - mw',
            'iw8_pi_mike9': 'Renetti', 'iw8_pi_mike1911': '1911',
            'iw8_pi_golf21': 'X16', 'iw8_pi_decho': '.50 GS',
            'iw8_pi_papa320': 'M19', 'iw8_me_riotshield': 'Riot Shield',
            'iw8_sm_mpapa7': 'MP7', 'iw8_sm_augolf': 'AUG - mw',
            'iw8_sm_papa90': 'P90', 'iw8_sm_charlie9': 'ISO',
            'iw8_sm_mpapa5': 'MP5', 'iw8_sm_smgolf45': 'Striker',
            'iw8_sm_beta': 'Bizon', 'iw8_sm_victor': 'Fennec',
            'iw8_sm_uzulu': 'Uzi', 'iw8_me_akimboblunt': 'Kali Sticks',
            'iw8_me_akimboblades': 'Katanas', 'iw8_knife': 'Knife',
            # Cold War
            "iw8_ar_t9accurate": "Krig", "iw8_ar_t9damage": "AK-47",
            "iw8_ar_t9fastfire": "FFAR", "iw8_ar_t9fasthandling": "Groza",
            "iw8_ar_t9longburst": "M16", "iw8_ar_t9mobility": "QBZ",
            "iw8_ar_t9standard": "XM4", "iw8_knife_t9loadout": "Knife",
            "iw8_la_t9freefire": "RPG-7 - cw", "iw8_la_t9grenadelauncher": "M79",
            "iw8_la_t9launcher": "M79", "iw8_la_t9standard": "Cigma",
            "iw8_lm_t9accurate": "Stoner", "iw8_lm_t9accuratecb": "Stoner 63a - Coverband",
            "iw8_lm_t9light": "RPD", "iw8_lm_t9slowfire": "M60",
            "iw8_me_t9loadout": "Knife", "iw8_me_t9sledgehammer": "Sledgehammer",
            "iw8_me_t9wakizashi": "Wakizashi", "iw8_pi_t9burst": "Diamatti",
            "iw8_pi_t9burstcb": "Diamatti - Coverband", "iw8_pi_t9revolver": "Magnum - cw",
            "iw8_pi_t9semiauto": "1911", "iw8_sh_t9fullauto": "Streetsweeper",
            "iw8_sh_t9pump": "Hauer", "iw8_sh_t9semiauto": "Gallo",
            "iw8_sh_t9semiautocb": "Gallo SA12 - Coverband", "iw8_sm_t9accurate": "LC10",
            "iw8_sm_t9burst": "KSP", "iw8_sm_t9capacity": "Bullfrog",
            "iw8_sm_t9fastfire": "MAC-10", "iw8_sm_t9handling": "Milano",
            "iw8_sm_t9heavy": "AK-74u", "iw8_sm_t9powerburst": "AUG - cw",
            "iw8_sm_t9powerburstcb": "Aug - Coverband", "iw8_sm_t9standard": "MP5",
            "iw8_sm_t9standardcb": "MP5 - Coverband", "iw8_sn_t9cannon": "NTW-20",
            "iw8_sn_t9damagesemi": "Type 63", "iw8_sn_t9powersemi": "M82",
            "iw8_sn_t9powersemicb": "Barret M82 - Coverband", "iw8_sn_t9precisionsemi": "DMR 14",
            "iw8_sn_t9quickscope": "Pelington", "iw8_sn_t9standard": "LW3",
            "iw8_fists": "Fists"
        }
        self.whole: pd.DataFrame = self.evaluate_df()
        self.last_match_date_time = list(self.whole['startDateTime'])[-1]
            
        if refresh:
            self.whole = refresh_data(old_df=self.whole,
                                      last_match_timestamp=self.last_match_date_time,
                                      repo=self.repo,
                                      filename=None,
                                      savedf=True)
        
        self.match_id_lst = set(self.whole['matchID'])
        self.name_uno_dict = self.get_match_id_set(self.whole)
        self.my_uno = list(set(self.whole[self.whole['username'] == 'Claim']['uno']))[0]
        self.our_df, self.other_df = self.get_our_and_other_df(self.whole)
    
    def evaluate_df(self,
                    link: str = 'Personal_Match_Data_v2.csv',
                    ) -> pd.DataFrame:
        
        df = pd.read_csv(self.repo + link, index_col='Unnamed: 0')
        start_time_lst = list(df['utcStartSeconds'])
        df['startDateTime'] = [datetime.datetime.utcfromtimestamp(i) for i in start_time_lst]
        end_time_lst = list(df['utcEndSeconds'])
        df['endDateTime'] = [datetime.datetime.utcfromtimestamp(i) for i in end_time_lst]
        day_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        start_time_timestamp_lst = list(df['startDateTime'])
        df['weekDay'] = [day_dic[i.weekday()] for i in start_time_timestamp_lst]
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
    
    # def evaluate_df(self,
    #                 link: str = None,
    #                 prebuilt: bool = True,
    #                 ):
    #
    #     cols = [
    #         # Match Related
    #         'dateTime', 'startDate', 'startTime', 'endDate', 'endTime', 'weekday', 'map', 'mode', 'matchID',
    #         'duration', 'playerCount',
    #         # identification
    #         'teamCount', 'teamPlacement', 'placementPercent', 'team', 'username', 'uno',
    #         # Kill related
    #         'kills', 'assists', 'deaths', 'headshots', 'kdRatio', 'executions',
    #         'longestStreak', 'damageDone', 'damageTaken', 'gulagDeaths', 'gulagKills',
    #         'wallBangs',
    #         # Score based
    #         'medalXp', 'matchXp', 'scoreXp', 'totalXp', 'score', 'scorePerMinute',
    #         'totalMissionXpEarned', 'totalMissionWeaponXpEarned',
    #         # Mission based
    #         'missionsComplete', 'objectiveTeamWiped',
    #         'objectiveLastStandKill', 'objectiveBrCacheOpen', 'objectiveMunitionsBoxTeammateUsed',
    #         'objectiveBrDownEnemyCircle3', 'objectiveBrDownEnemyCircle2',
    #         'objectiveBrMissionPickupTablet', 'objectiveBrKioskBuy', 'missions',
    #         # Telometrics
    #         'distanceTraveled', 'percentTimeMoving', 'teamSurvivalTime', 'timePlayed',
    #         # Weapon related
    #         'tactical', 'lethal', 'primaryWeaponName', 'primaryWeaponAttachments',
    #         'secondaryWeaponName', 'secondaryWeaponAttachments',
    #     ]
    #
    #     if prebuilt:
    #         df_open = pd.read_csv(self.repo + link, index_col='Unnamed: 0')
    #         df_open['dateTime'] = [datetime.datetime.strptime(i, '%Y-%m-%d %H:%M') for i in df_open['dateTime']]
    #         return df_open.sort_values(by='dateTime').reset_index(drop=True)[cols]
    #
    #     else:
    #         def convert_to_date_or_time(text, index: int = None, str_output: bool = True):
    #             if str_output:
    #                 return str(datetime.datetime.fromtimestamp(text).strftime('%Y-%m-%d %H:%M').split(" ", 1)[index])
    #             else:
    #                 return datetime.datetime.strptime(text, "%Y-%m-%d")
    #
    #         def convert_to_str(text):
    #             return str(text)
    #
    #         dfn = pd.read_csv(self.repo + link, index_col='Unnamed: 0')
    #         column_lst = ['map', 'mode', 'matchID', 'duration', 'playerCount', 'teamCount', 'kills', 'medalXp',
    #                       'objectiveTeamWiped', 'objectiveLastStandKill', 'matchXp', 'scoreXp', 'wallBangs', 'score',
    #                       'totalXp',
    #                       'headshots', 'assists', 'challengeXp', 'scorePerMinute', 'distanceTraveled',
    #                       'teamSurvivalTime',
    #                       'deaths', 'objectiveMunitionsBoxTeammateUsed', 'objectiveBrDownEnemyCircle3', 'kdRatio',
    #                       'objectiveBrDownEnemyCircle2', 'objectiveBrMissionPickupTablet', 'bonusXp',
    #                       'objectiveBrKioskBuy',
    #                       'gulagDeaths', 'timePlayed', 'executions', 'gulagKills', 'objectiveBrCacheOpen', 'miscXp',
    #                       'longestStreak', 'teamPlacement', 'damageDone', 'damageTaken', 'team', 'username', 'uno',
    #                       'missionsComplete', 'totalMissionXpEarned', 'totalMissionWeaponXpEarned',
    #                       ]
    #
    #         for col in column_lst:
    #             dfn[col] = [convert_to_str(i) for i in dfn[col]]
    #
    #         for gun in ['primaryWeapon', 'secondaryWeapon']:
    #             gun_name_lst = []
    #             attachments = []
    #             for i in dfn[gun]:
    #                 temp_dict = literal_eval(i)
    #                 if temp_dict != 0:
    #                     gun_name_lst.append(temp_dict['name'])
    #                     attachments.append(
    #                         ' '.join([i['name'] for i in temp_dict['attachments'] if i['name'] is not None]))
    #                 else:
    #                     gun_name_lst.append(' ')
    #                     attachments.append(' ')
    #             dfn[gun + 'Name'] = gun_name_lst
    #             dfn[gun + 'Attachments'] = attachments
    #
    #         for perk in ['perks', 'extraPerks']:
    #             perk_lst = []
    #             for i in dfn[perk]:
    #                 temp_dict = literal_eval(i)
    #                 if temp_dict != 0:
    #                     perk_lst.append(' '.join([i['name'] for i in temp_dict if i['name'] is not None]))
    #                 else:
    #                     perk_lst.append(' ')
    #             dfn[perk] = perk_lst
    #
    #         for equip in ['tactical', 'lethal']:
    #             equip_lst = []
    #             for i in dfn[equip]:
    #                 temp_dict = literal_eval(i)
    #                 if temp_dict != 0:
    #                     equip_lst.append(temp_dict['name'])
    #                 else:
    #                     equip_lst.append(' ')
    #             dfn[equip] = equip_lst
    #
    #         mission_lst = []
    #         for mission in df['missionStatsByType']:
    #             temp_dict = literal_eval(mission)
    #             missions = ' '
    #             if temp_dict != 0:
    #                 missions = ' '.join([i + str(int(temp_dict[i]['count'])) for i in temp_dict.keys()])
    #             mission_lst.append(missions)
    #         dfn['missions'] = mission_lst
    #
    #         dfn['dateTime'] = (dfn['date'] + ' ' + dfn['time']).astype(str)
    #         dfn['dateTime'] = [datetime.datetime.strptime(i, '%Y-%m-%d %H:%M') for i in dfn['dateTime']]
    #         dfn['startDate'] = [convert_to_date_or_time(i, 0) for i in dfn['utcStartSeconds']]
    #         dfn['startTime'] = [convert_to_date_or_time(i, 1) for i in dfn['utcStartSeconds']]
    #         dfn['endDate'] = [convert_to_date_or_time(i, 0) for i in dfn['utcEndSeconds']]
    #         dfn['endTime'] = [convert_to_date_or_time(i, 1) for i in dfn['utcEndSeconds']]
    #         dfn['percentTimeMoving'] = [str(round(i, 2)) for i in dfn['percentTimeMoving']]
    #         day_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday',
    #                    6: 'Sunday'}
    #         dfn['weekday'] = [day_dic[i.date().weekday()] for i in dfn['date']]
    #         dfn['placementPercent'] = (1 - dfn['teamPlacement'] / dfn['teamCount']).round(2)
    #
    #         new_cols = ['startDate', 'startTime', 'endDate', 'endTime', 'weekday', 'percentTimeMoving',
    #                     'primaryWeaponName',
    #                     'primaryWeaponAttachments', 'secondaryWeaponName', 'secondaryWeaponAttachments', 'perks',
    #                     'extraPerks',
    #                     'tactical', 'lethal', 'missions']
    #
    #         dfn_output = dfn[column_lst + new_cols]
    #         dfn_output.to_csv(self.repo + link.split('.')[0] + '_Prebuilt.' + link.split('.')[1])
    #         df_open = pd.read_csv(self.repo + link.split('.')[0] + '_Prebuilt.' + link.split('.')[1],
    #                               index_col='Unnamed: 0')
    #         df_open['dateTime'] = [datetime.datetime.strptime(i, '%Y-%m-%d %H:%M') for i in df_open['dateTime']]
    #
    #         large_damage_taken = list(df_open[df_open['damageTaken'] > 100000].index)
    #         for i in large_damage_taken:
    #             df_open.loc[i, 'damageTaken'] = df_open.loc[i, 'damageDone']
    #
    #         return df_open.sort_values(by='dateTime').reset_index(drop=True)[cols]
    
    @staticmethod
    def get_match_id_set(df: pd.DataFrame) -> dict:
        temp = df[['uno', 'username']].copy().drop_duplicates(subset=['uno', 'username'],
                                                              keep='first',
                                                              ignore_index=True)
        return {j['username']: j['uno'] for i, j in temp.iterrows()}
    
    def get_our_and_other_df(self,
                             df: pd.DataFrame,
                             name: str = None,
                             ):
        if name:
            uno = self.name_uno_dict[name]
        else:
            uno = self.my_uno
            
        mid, team = list(df[df['uno'] == uno]['matchID']), list(df[df['uno'] == uno]['team'])
        our_match_team = {match: team[ind] for ind, match in enumerate(mid)}
        our_lst = sum([[i for i in df[(df['matchID'] == ind) & (df['team'] == our_match_team[ind])].index] for ind in
                       our_match_team.keys()], [])
        whole_index = df.index
        other_lst = [i for i in whole_index if i not in our_lst]
        return df.iloc[our_lst], df.iloc[other_lst]
    
    @staticmethod
    def get_placement_data(df: pd.DataFrame,
                           place,
                           map_specfic: str = None,
                           ) -> pd.DataFrame:
        # mp_escape or mp_don
        
        if type(place) == int:
            if map_specfic:
                return df[(df['teamPlacement'] == place) & (df['map'] == map_specfic)]
            else:
                return df[df['teamPlacement'] == place]
        else:
            if map_specfic:
                return df[
                    (place[0] <= df['teamPlacement']) & (df['teamPlacement'] <= place[1]) & (df['map'] == map_specfic)]
            else:
                return df[(place[0] <= df['teamPlacement']) & (df['teamPlacement'] <= place[1])]
    
    @staticmethod
    def regress(xtrain,
                ytrain,
                xtest,
                ytest,
                test=False,
                plotn=False
                ):
        
        X = add_constant(xtrain)
        result = np.linalg.lstsq(X, ytrain, rcond=None)
        alpha, beta = result[0]
        
        # coef = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(ytrain)
        r2 = np.corrcoef(xtrain, ytrain)[0, 1] ** 2  # r^2
        
        y_hat = beta * xtrain + alpha
        res = ytrain - y_hat
        var_beta_hat = np.linalg.inv(X.T @ X) * ((res.T @ res) / (len(xtrain) - 2))
        std_error = tuple(np.diag(var_beta_hat) ** .5)
        t, p = stats.ttest_ind(ytrain, xtrain, equal_var=False)  # t test and p value
        p_m = stats.norm.ppf(.95) * (np.std(xtrain) / np.sqrt(len(xtrain)))
        conf = (np.mean(xtrain) - p_m, np.mean(xtrain) + p_m)
        
        x_fit = np.linspace(np.floor(xtrain.min()), np.ceil(xtrain.max()), 2)
        y_fit = alpha * x_fit + beta
        
        if plotn:
            plt.scatter(xtrain, ytrain, marker='o')
            plt.plot(x_fit, y_fit, 'r--')
            plt.title('Training Best-fit linear model')
            plt.show()
        
        if test:
            pred = np.dot(add_constant(xtest), result[0])
            resid = np.dot(add_constant(xtest), result[0]) - ytest
            mse = np.mean(np.square(ytest - xtest))
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(xtest - ytest))
            
            if plotn:
                plt.scatter(pred, resid, marker='o')
                plt.title('Test Best-fit linear model')
                plt.show()
            
            print('Intercept, Beta, R2, Standard Error, T-Statistic, P-Values, Confidence Inter')
            print('MSE, RMSE, MAE')
            print('Predicted, Residuals')
            return (((alpha, beta), r2, std_error, t, p, conf), (mse, rmse, mae), (pred, resid))
        
        else:
            print('Intercept, Beta, R2, Standard Error, T-Statistic, P-Values, Confidence Inter')
            return ((alpha, beta), r2, std_error, t, p, conf)
    
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
    
    @staticmethod
    def get_data(df,
                 cols: list,
                 team: bool = True,
                 ):
        
        ind = set(df['matchID'])
        if team:
            temp_lst_final = [
                [list(df[df['matchID'] == i]['dateTime'])[0], i] +
                [np.around(np.mean(df[df['matchID'] == i][col]), 2) for col in cols] for i in ind]
            return pd.DataFrame(temp_lst_final,
                                columns=['dateTime', 'matchID'] + cols).sort_values('dateTime').reset_index(drop=True)
        else:
            return df[['dateTime', 'matchID'] + cols]
    
    def specific_weapon(self,
                        name: str,
                        weapon_name: str = None,
                        map_specfic: str = None,
                        cols: list = None,
                        ):
        
        uno = self.name_uno_dict[name]
        name_df = self.whole[self.whole['uno'] == uno]
        
        if map_specfic is not None:
            name_df = name_df[name_df['map'] == map_specfic]
        
        actv_name = list(self.gun_dic.keys())
        actual_name = list(self.gun_dic.values())
        
        if weapon_name is None:
            print(), print(actual_name), print()
            weapon_name = input()
        
        gun = actv_name[actual_name.index(weapon_name)]
        
        if cols is None:
            cols = ['kills', 'deaths', 'kd']
        
        result = name_df[(name_df['pw_gun'] == gun) | (name_df['sw_gun'] == gun)].set_index('date time').sort_index()
        
        if ('kd' in cols) | ('kdRatio' in cols):
            result['kd'] = (result['kills'] / result['deaths']).round(2)
        
        final = result[cols]
        
        return final
    
    def weapon_results(self,
                       name: str,
                       map_specfic: str = None,
                       sort_by: str = None,
                       limit: int = None,
                       threshold: list = None,
                       specfic_weapon: str or list = None,
                       compute_cumsum: bool = False,
                       ):
        
        uno = self.name_uno_dict[name]
        name_df = self.whole[self.whole['uno'] == uno]
        
        if map_specfic is not None:
            name_df = name_df[name_df['map'] == map_specfic]
        
        if specfic_weapon is None:
            guns_used = set(list(name_df['pw_gun']) + list(name_df['sw_gun']))
            temp_gun_dict = {}
            for gun in guns_used:
                if gun != '0':
                    pw_lst = name_df[(name_df['pw_gun'] == gun) | (name_df['sw_gun'] == gun)][['kills', 'deaths']]
                    total_sum = list(np.sum(pw_lst, axis=0))
                    
                    temp_gun_dict[gun] = list([total_sum[0],
                                               total_sum[1],
                                               total_sum[0] / total_sum[1],
                                               set(pw_lst.index)
                                               ])
            
            gun_dict_keys = {i: True for i in self.gun_dic.keys()}
            final_dict = {}
            for i in temp_gun_dict.keys():
                if i in gun_dict_keys:
                    final_dict[self.gun_dic[i]] = temp_gun_dict[i]
                else:
                    final_dict[i] = temp_gun_dict[i]
            
            result = pd.DataFrame.from_dict(final_dict, orient='index').reset_index()
            result.columns = ['gun', 'kills', 'deaths', 'kd', 'index']
            
            if sort_by is not None:
                result = result.sort_values(sort_by, ascending=False).reset_index(drop=True)
            
            if threshold is not None:
                result = result[result[threshold[0]] >= threshold[1]].reset_index(drop=True)
            
            if limit is not None:
                result = result.iloc[:limit, :]
        
        else:
            if type(specfic_weapon) == str:
                result = self.specific_weapon(name=name,
                                              weapon_name=specfic_weapon,
                                              map_specfic=map_specfic,
                                              cols=None
                                              )
                return result
            else:
                w = {}
                for weapon in specfic_weapon:
                    sw = self.specific_weapon(name=name,
                                              weapon_name=weapon,
                                              map_specfic=map_specfic,
                                              cols=None
                                              )
                    sw.columns = [i + '_' + weapon for i in sw.columns]
                    w[weapon] = sw
                
                wn = list(w.values())
                index, cols = [], []
                for i, j in enumerate(wn):
                    index += wn[i].index
                    cols += list(wn[i].columns)
                
                final_df = pd.DataFrame(index=index, columns=cols)
                for i, j in enumerate(wn):
                    for k in j.columns:
                        if 'kd' not in k:
                            if compute_cumsum is not None:
                                final_df[k] = j[k].cumsum()
                            else:
                                final_df[k] = j[k]
                        else:
                            final_df[k] = j[k]
                
                final_df = final_df.sort_index()
                
                if compute_cumsum is not None:
                    result = final_df.ffill().fillna(0)
                else:
                    result = final_df.fillna(0)
        
        return result
    
    @staticmethod
    def regression_calcs(df: pd.DataFrame,
                         y: np.ndarray,
                         r2: float = None,
                         constant_coef: float = None,
                         variable_coef: float = None,
                         confidence_interval_lower: float = None,
                         confidence_interval_higher: float = None,
                         p_value: float = 0.05,
                         sort_value: list = None,
                         filtering: bool = False,
                         ):
        
        if sort_value is None:
            sort_value = ['P Value', True]
        
        final_cols = []
        if r2 is not None:
            final_cols.append('R Squared')
        
        if constant_coef is not None:
            final_cols.append('Constant Coefficient')
        
        if variable_coef is not None:
            final_cols.append('Word Coefficient')
        
        if confidence_interval_lower is not None:
            final_cols.append('Lower Confidence Interval')
        
        if confidence_interval_higher is not None:
            final_cols.append('Higher Confidence Interval')
        
        if p_value is not None:
            final_cols.append('P Value')
        
        col_lst = list(df.columns)
        result = {}
        for word in col_lst:
            x = add_constant(np.array(df[word], dtype=float))
            model = regression.linear_model.OLS(y, x).fit()
            
            model_results = []
            if r2 is not None:
                model_results.append(model.rsquared)
            
            if constant_coef is not None:
                model_results.append(model.params[0])
            
            if variable_coef is not None:
                model_results.append(model.params[1])
            
            if confidence_interval_lower is not None:
                model_results.append(model.conf_int()[1, :2][0])
            
            if confidence_interval_higher is not None:
                model_results.append(model.conf_int()[1, :2][1])
            
            if p_value is not None:
                model_results.append(model.pvalues[1])
            
            result[word] = model_results
        
        result_df = pd.DataFrame.from_dict(result, orient='index', columns=final_cols)
        if filtering:
            if r2 is not None:
                result_df = result_df[result_df['R Squared'] > r2]
    
            if constant_coef is not None:
                result_df = result_df[result_df['Constant Coefficient'] > constant_coef]
        
            if variable_coef is not None:
                result_df = result_df[result_df['Word Coefficient'] > variable_coef]
        
            if confidence_interval_lower is not None:
                result_df = result_df[result_df['Lower Confidence Interval'] > confidence_interval_lower]
        
            if confidence_interval_higher is not None:
                result_df = result_df[result_df['Higher Confidence Interval'] > confidence_interval_higher]
        
            if p_value is not None:
                result_df = result_df[result_df['P Value'] < p_value]
        
        final = result_df.sort_values(sort_value[0], ascending=sort_value[1]).round(3)
        
        return final, list(final.index)

    def get_daily_hourly_weekday_stats(self,
                                       person: str = None,
                                       map: str = 'mp_escape',
                                       save: bool = False,
                                       combined_item: str = 'kdRatio',
                                       combined_method: str = 'mean',
                                       ):
        
        def daily_stats(_df: pd.DataFrame,
                        _map: str = 'mp_escape',
                        ):
        
            dfn = _df.iloc[[i for i, j in enumerate(list(_df['map'])) if _map in str(j)]]
            _matches = set(list(dfn['matchID']))
            if len(_matches) > 0:
                wins, top_fives = [], []
                for match in _matches:
                    temp = dfn[dfn['matchID'] == match]
                    place = int(list(temp['teamPlacement'])[0])
                
                    if place == 1:
                        wins.append(1)
                    elif (1 < place) & (place >= 5):
                        top_fives.append(1)
                    else:
                        wins.append(0)
                        top_fives.append(0)
            
                return [int(dfn['kills'].sum()), int(dfn['deaths'].sum()), sum(wins), sum(top_fives), len(_matches), dfn['teamPlacement'].mean()]
            else:
                return [0, 0, 0, 0, 0, 0]

        days_lst = self.our_df['startDate'].unique()
        daily_info = pd.DataFrame.from_dict(
            {day: daily_stats(_df=self.our_df[self.our_df['startDate'] == day],
                              _map=map) for day in days_lst},
            orient='index',
            columns=['dailyKills', 'dailyDeaths', 'dailyWins', 'dailyTopFives', 'dailyMatchCount',
                     'dailyAverageTeamPlacement']
        )
        daily_info = daily_info[daily_info['dailyMatchCount'] > 0]
        daily_info['dailyKD'] = (daily_info['dailyKills'] / daily_info['dailyDeaths']).round(2)

        hours = range(24)
        hours_lst = list(self.our_df['startTime'])
        hourly_info = pd.DataFrame.from_dict(
            {hour: daily_stats(
                _df=self.our_df.iloc[[i for i, j in enumerate(hours_lst) if hour == int(str(j).split(':')[0])]],
                _map=map) for hour in hours},
            orient='index',
            columns=['hourlyKills', 'hourlyDeaths', 'hourlyWins', 'hourlyTopFives', 'hourlyMatchCount',
                     'hourlyAverageTeamPlacement']
        )
        hourly_info['hourlyKD'] = (hourly_info['hourlyKills'] / hourly_info['hourlyDeaths']).fillna(0).round(2)
        hourly_info.index = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00',
                             '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00',
                             '20:00', '21:00', '22:00', '23:00']

        weekdays_lst = self.our_df['weekday'].unique()
        weekday_info = pd.DataFrame.from_dict(
            {weekday: daily_stats(_df=self.our_df[self.our_df['weekday'] == weekday],
                                  _map=map) for weekday in weekdays_lst},
            orient='index',
            columns=['weekDayKills', 'weekDayDeaths', 'weekDayWins', 'weekDayTopFives', 'weekDayMatchCount',
                     'weekDayAverageTeamPlacement']
        )
        weekday_info = weekday_info[weekday_info['weekDayMatchCount'] > 0]
        weekday_info['dailyKD'] = (weekday_info['weekDayKills'] / weekday_info['weekDayDeaths']).round(2)
        weekday_info.index = [str(i) for i in weekday_info.index]
        weekday_info = weekday_info.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
                                             'Sunday'])
        # day_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        # hours = range(24)
        # dic = {}
        # for weekday in day_dic.values():
        #     dfn = cod.our_df[cod.our_df['weekDay'] == weekday].reset_index(drop=True)
        #     temp_dic = {}
        #     for _hour in hours:
        #         lst = []
        #         for i, j in enumerate(dfn['startDateTime']):
        #             if j.hour == _hour:
        #                 lst.append(dfn.loc[i, 'kills'])
        #         temp_dic[_hour] = sum(lst)
        #     dic[weekday] = temp_dic
        #
        # temp_df = pd.DataFrame.from_dict(dic)
        # temp_df.index = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00',
        #                  '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00',
        #                  '20:00', '21:00', '22:00', '23:00']
        return daily_info, hourly_info, weekday_info

    def get_person_data(self,
                        person_lst: list,
                        map: str = 'mp_escape',
                        ) -> pd.DataFrame:
        
        def _get_data(_person: str = None,
                      _map: str = 'mp_escape'
                      ):
    
            _df = self.our_df[self.our_df['uno'] == self.name_uno_dict[_person]]
            _df = _df.iloc[[i for i, j in enumerate(list(_df['map'])) if _map in str(j)]]
    
            return [_df['kills'].sum(), _df['deaths'].sum(), _df['damageDone'].sum(), _df['damageTaken'].sum(),
                    _df['kdRatio'].mean(), _df['scorePerMinute'].mean(), _df['teamPlacement'].mean(),
                    _df['damageDone'].mean(), _df['damageTaken'].mean(), _df['distanceTraveled'].mean(),
                    _df['percentTimeMoving'].mean(), _df['timePlayed'].mean()]
        
        dfn = pd.DataFrame.from_dict({name: _get_data(name, _map=map) for name in person_lst},
                                     orient='index',
                                     columns=['killsTotal', 'deathsTotal', 'damageDoneTotal', 'damageTakenTotal',
                                              'kdAverage', 'scorePerMinuteAverage', 'placementAverage',
                                              'damageDoneAverage', 'damageTakenAverage', 'distanceTraveledAverage',
                                              'percentTimeMovingAverage', 'timePlayedAverage'])
        return dfn

if __name__ == '__main__':
    start_timen = time.time()
    cod = CallofDuty(refresh=False)
    print(''), print('Data Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))

    # daily_data, hourly_data, weekday_data = cod.get_daily_hourly_weekday_stats()
    # person_info = cod.get_person_data(person_lst=['Claim', 'MONEYMIKE0410', 'LeoxGemini', 'TheKing109', 'Rhino5378',
    #                                               'spectator95'])
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
