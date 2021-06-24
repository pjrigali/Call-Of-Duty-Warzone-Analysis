import pandas as pd
from Utils.gun_dictionary import gun_dict


def get_person_data(person_lst: list,
                    data: pd.DataFrame,
                    uno_dict: dict,
                    map_choice: str = 'mp_escape',
                    ) -> pd.DataFrame:
    
    def _get_data(_person: str = None,
                  _data: pd.DataFrame = data,
                  _uno_dict: dict = None,
                  _map: str = 'mp_escape'
                  ):
        
        _df = _data[_data['uno'] == _uno_dict[_person]]
        _df = _df.iloc[[i for i, j in enumerate(list(_df['map'])) if _map in str(j)]]
        
        return [_df['kills'].sum(), _df['deaths'].sum(), _df['damageDone'].sum(), _df['damageTaken'].sum(),
                _df['kdRatio'].mean(), _df['scorePerMinute'].mean(), _df['teamPlacement'].mean(),
                _df['damageDone'].mean(), _df['damageTaken'].mean(), _df['distanceTraveled'].mean(),
                _df['percentTimeMoving'].mean(), _df['timePlayed'].mean()]
    
    dfn = pd.DataFrame.from_dict({name: _get_data(_person=name,
                                                  _data=data,
                                                  _uno_dict=uno_dict,
                                                  _map=map_choice) for name in person_lst},
                                 orient='index',
                                 columns=['killsTotal', 'deathsTotal', 'damageDoneTotal', 'damageTakenTotal',
                                          'kdAverage', 'scorePerMinuteAverage', 'placementAverage',
                                          'damageDoneAverage', 'damageTakenAverage', 'distanceTraveledAverage',
                                          'percentTimeMovingAverage', 'timePlayedAverage'])
    return dfn


def get_daily_hourly_weekday_stats(person: str,
                                   data: pd.DataFrame,
                                   map_choice: str = 'mp_escape',
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
            
            return [int(dfn['kills'].sum()), int(dfn['deaths'].sum()), sum(wins), sum(top_fives), len(_matches),
                        dfn['teamPlacement'].mean()]
        else:
            return [0, 0, 0, 0, 0, 0]
    
    days_lst = data['startDate'].unique()
    daily_info = pd.DataFrame.from_dict(
        {day: daily_stats(_df=data[data['startDate'] == day],
                          _map=map_choice) for day in days_lst},
        orient='index',
        columns=['dailyKills', 'dailyDeaths', 'dailyWins', 'dailyTopFives', 'dailyMatchCount',
                 'dailyAverageTeamPlacement']
        )
    daily_info = daily_info[daily_info['dailyMatchCount'] > 0]
    daily_info['dailyKD'] = (daily_info['dailyKills'] / daily_info['dailyDeaths']).round(2)
    
    hours = range(24)
    hours_lst = list(data['startTime'])
    hourly_info = pd.DataFrame.from_dict(
        {hour: daily_stats(
            _df=data.iloc[[i for i, j in enumerate(hours_lst) if hour == int(str(j).split(':')[0])]],
            _map=map_choice) for hour in hours},
        orient='index',
        columns=['hourlyKills', 'hourlyDeaths', 'hourlyWins', 'hourlyTopFives', 'hourlyMatchCount',
                 'hourlyAverageTeamPlacement']
    )
    hourly_info['hourlyKD'] = (hourly_info['hourlyKills'] / hourly_info['hourlyDeaths']).fillna(0).round(2)
    hourly_info.index = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00',
                         '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00',
                         '20:00', '21:00', '22:00', '23:00']
    
    weekdays_lst = data['weekDay'].unique()
    weekday_info = pd.DataFrame.from_dict(
        {weekday: daily_stats(_df=data[data['weekDay'] == weekday],
                              _map=map_choice) for weekday in weekdays_lst},
        orient='index',
        columns=['weekDayKills', 'weekDayDeaths', 'weekDayWins', 'weekDayTopFives', 'weekDayMatchCount',
                     'weekDayAverageTeamPlacement']
    )
    weekday_info = weekday_info[weekday_info['weekDayMatchCount'] > 0]
    weekday_info['dailyKD'] = (weekday_info['weekDayKills'] / weekday_info['weekDayDeaths']).round(2)
    weekday_info.index = [str(i) for i in weekday_info.index]
    weekday_info = weekday_info.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday'])
    
    # day_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    # hours = range(24)
    # dic = {}
    # for weekday in day_dic.values():
    #     dfn = data[data['weekDay'] == weekday].reset_index(drop=True)
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
    
    if save:
        daily_info.to_csv('daily_info.csv')
        hourly_info.to_csv('hourly_info.csv')
        weekday_info.to_csv('weekday_info.csv')
    
    return daily_info, hourly_info, weekday_info


def get_weapons(data: pd.DataFrame,
                person: str = None,
                uno_dict: dict = None,
                map_choice: str = None,
                columns: list = None,
                sort_by: str = None,
                save: bool = False,
                ) -> pd.DataFrame:
    
    if person:
        data = data[data['uno'] == uno_dict[person]].copy()
    
    if map_choice:
        data = data.iloc[[i for i, j in enumerate(data['map']) if map_choice in j]].copy()
        
    if columns is None:
        columns = ['kills', 'deaths', 'headshots', 'assists']
        
    if sort_by is None:
        sort_by = columns[0]

    col_lst = [col for col in data.columns if 'yWeapon' in col and 'Attach' not in col]
    gun_set = set(sum([list(data[col]) for col in col_lst], []))
    gun_dict_n = {gun: [0, 0, 0, 0] for gun in gun_set}
    for col in col_lst:
        for gun in gun_set:
            temp_n = list(data[data[col] == gun][columns].sum())
            gun_dict_n[gun] = [gun_dict_n[gun][0] + temp_n[0],
                               gun_dict_n[gun][1] + temp_n[1],
                               gun_dict_n[gun][2] + temp_n[0],
                               gun_dict_n[gun][3] + temp_n[3]]

    gun_df = pd.DataFrame.from_dict(gun_dict_n, orient='index', columns=columns).sort_values(sort_by, ascending=False)
    gun_df['kd'] = gun_df['kills'] / gun_df['deaths']
    gun_df.index = [gun_dict[gun] if gun in gun_dict.keys() else gun for gun in gun_df.index]
    
    if save:
        gun_df.to_csv('weapon_info.csv')
    
    return gun_df
