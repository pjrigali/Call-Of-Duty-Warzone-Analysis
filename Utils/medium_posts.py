from typing import List
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from document_filter import DocumentFilter
from base import running_mean, cumulative_mean, normalize


def find_hackers_from_hacker_df(hacker_doc_filter: DocumentFilter, our_doc_filter: DocumentFilter) -> pd.DataFrame:
    our_data = our_doc_filter.df
    data = hacker_doc_filter.df
    col_lst = ['headshots', 'kills', 'deaths', 'kdRatio', 'scorePerMinute', 'distanceTraveled',
               'objectiveBrKioskBuy', 'percentTimeMoving', 'longestStreak', 'damageDone', 'damageTaken',
               'missionsComplete', 'objectiveLastStandKill', 'objectiveBrDownEnemyCircle1',
               'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3', 'objectiveBrDownEnemyCircle4',
               'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6', 'objectiveTeamWiped', 'objectiveReviver',
               'objectiveMunitionsBoxTeammateUsed', 'objectiveBrCacheOpen', 'objectiveMedalScoreKillSsRadarDrone']
    whole_df = pd.DataFrame()
    hacker_data_dic = {}
    our_data_dic = {}
    for _map in ['mp_e', 'mp_d']:
        data_n = data.iloc[[i for i, j in enumerate(list(data['map'])) if _map in str(j)]]
        our_data_n = our_data.iloc[[i for i, j in enumerate(list(our_data['map'])) if _map in str(j)]]
        data_n['username'] = [i.lower() for i in list(data_n['username'])]

        hacker_dic = {i: 0 for i in set(data_n['uno'])}
        for i in data_n['uno']:
            hacker_dic[str(i)] += 1
        hacker_df_count = pd.DataFrame.from_dict(hacker_dic, orient='index')
        hacker_df_count = hacker_df_count[hacker_df_count[0] > 4]
        hacker_df_count.index = [list(data_n[data_n['uno'] == str(i)]['username'])[0] for i in list(hacker_df_count.index)]
        hacker_name_uno_dic = {_name: set(data_n[data_n['username'] == _name]['uno']) for _name in set(hacker_df_count.index)}

        dic = {}
        for _name in hacker_name_uno_dic.keys():
            uno_lst = hacker_name_uno_dic[_name]
            val_dic = {col: [] for col in col_lst}
            for uno in uno_lst:
                temp = data_n[data_n['uno'] == uno].fillna(0.0)
                for col in col_lst:
                    val_dic[col] = val_dic[col] + list(temp[col])
            dic[_name] = {col: np.mean(val_dic[col]) for col in col_lst}

        temp_df = pd.DataFrame.from_dict(dic, orient='index', columns=col_lst).fillna(0.0)
        temp_df = temp_df[temp_df['kdRatio'] > np.floor(np.mean(our_data_n['kdRatio']) + np.std(our_data_n['kdRatio'],
                                                                                              ddof=1) * 2)]
        # temp_df['count'] = [np.sum(hacker_df_count.loc[_name]) for _name in list(temp_df.index)]
        temp_df['headshotRatio'] = temp_df['headshots'] / temp_df['kills']
        hacker_data_dic[_map] = temp_df

        our_name_uno_dic = {_name: our_doc_filter.username_dic[_name] for _name in our_doc_filter.username_lst}
        dic = {}
        for _name in our_name_uno_dic.keys():
            temp = our_data_n[our_data_n['uno'] == our_name_uno_dic[_name]].fillna(0.0)
            dic[_name] = {col: np.mean(temp[col]) for col in col_lst}
        our_temp_df = pd.DataFrame.from_dict(dic, orient='index', columns=col_lst).fillna(0.0)
        our_final_df = pd.DataFrame([list(our_temp_df.mean())], columns=col_lst, index=['* squad mu'])
        our_final_df['headshotRatio'] = our_final_df['headshots'] / our_final_df['kills']
        our_data_dic[_map] = {i: our_final_df.loc['* squad mu', i] for i in our_final_df.columns}

        final_df = pd.concat([temp_df, our_final_df]).fillna(0.0)
        final_df['map'] = _map
        whole_df = pd.concat([whole_df, final_df])

    # Base
    # col_lst_above = ['deaths', 'objectiveBrKioskBuy', 'damageTaken', 'missionsComplete',
    #                  'objectiveMunitionsBoxTeammateUsed', 'objectiveBrCacheOpen',
    #                  'objectiveMedalScoreKillSsRadarDrone']
    # col_lst_below = ['headshots', 'kills', 'kdRatio', 'scorePerMinute', 'distanceTraveled', 'percentTimeMoving',
    #                  'longestStreak', 'damageDone', 'objectiveLastStandKill', 'objectiveBrDownEnemyCircle1',
    #                  'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3', 'objectiveBrDownEnemyCircle4',
    #                  'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6', 'objectiveTeamWiped',
    #                  'objectiveReviver', 'headshotRatio']

    # Tested
    # col_dic = {'mp_d':
    #                {'above': ['deaths', 'objectiveBrKioskBuy', 'missionsComplete',
    #                           'objectiveMedalScoreKillSsRadarDrone'],
    #                 'below': ['headshots', 'kills', 'kdRatio', 'scorePerMinute', 'percentTimeMoving',
    #                           'longestStreak', 'damageDone', 'objectiveLastStandKill', 'objectiveBrDownEnemyCircle1',
    #                           'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3',
    #                           'objectiveBrDownEnemyCircle4', 'objectiveBrDownEnemyCircle5',
    #                           'objectiveBrDownEnemyCircle6', 'objectiveTeamWiped', 'headshotRatio',
    #                           'objectiveBrCacheOpen']},
    #            'mp_e':
    #                {'above': ['deaths', 'objectiveBrKioskBuy', 'damageTaken', 'missionsComplete',
    #                           'objectiveMunitionsBoxTeammateUsed', 'objectiveBrCacheOpen',
    #                           'objectiveMedalScoreKillSsRadarDrone'],
    #                 'below': ['headshots', 'kills', 'kdRatio', 'scorePerMinute', 'distanceTraveled',
    #                           'percentTimeMoving', 'longestStreak', 'damageDone', 'objectiveLastStandKill',
    #                           'objectiveBrDownEnemyCircle1', 'objectiveBrDownEnemyCircle2',
    #                           'objectiveBrDownEnemyCircle3', 'objectiveBrDownEnemyCircle5',
    #                           'objectiveBrDownEnemyCircle6', 'objectiveTeamWiped', 'objectiveReviver', 'headshotRatio']}
    #            }
    #
    # hacker_lst = []
    # data = data.fillna(0.0)
    # for i, j in enumerate(data['kills']):
    #     temp = data.iloc[i]
    #
    #     if 'mp_e' in temp['map']:
    #         _map = 'mp_e'
    #     elif 'mp_d' in temp['map']:
    #         _map = 'mp_d'
    #     else:
    #         hacker_lst.append(0.0)
    #         continue
    #
    #     vals_1 = [1 if our_data_dic[_map][col] > temp[col] else 0 for col in col_dic[_map]['above']]
    #     vals_2 = [1 if our_data_dic[_map][col] < temp[col] else 0 for col in col_dic[_map]['below']]
    #     hacker_lst.append(np.mean(np.nan_to_num(vals_1 + vals_2)))
    # data['hackerProb'] = hacker_lst
    return whole_df


def deaths_per_circle(doc_filter: DocumentFilter) -> pd.DataFrame:
    data = doc_filter.df
    circle_lst = ['objectiveBrDownEnemyCircle1', 'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3',
                  'objectiveBrDownEnemyCircle4', 'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6']
    lst = []
    for _id in doc_filter.unique_ids:
        temp_df = data[data['matchID'] == _id].fillna(0.0)
        # temp_dic = {circle: temp_df[circle].sum() for circle in circle_lst}
        temp_dic = {'kills': temp_df['kills'].sum(),
                    'deaths': temp_df['deaths'].sum(),
                    'playerCount': temp_df['playerCount'].mean()}
        down_count = []
        for circle in circle_lst:
            val = temp_df[circle].sum()
            temp_dic[circle] = val
            down_count.append(val)

        if doc_filter.map_choice == 'mp_d':
            if sum(down_count) < 100:
                continue
            else:
                lst.append(temp_dic)
        else:
            if sum(down_count) == 0:
                continue
            else:
                lst.append(temp_dic)
    t = pd.DataFrame(lst, columns=['kills', 'deaths', 'playerCount'] + circle_lst)

    mu_playercount = round(t['playerCount'].mean(), 0)
    down_total_lst = []
    percent_revive_lst = []
    for row in range(len(t)):
        temp = t.iloc[row]
        down_total = []
        for circle in circle_lst:
            down_total.append(temp[circle])
        down_total_lst.append(np.sum(down_total))
        if doc_filter.map_choice == 'mp_d':
            percent_revive_lst.append(np.sum(down_total) / temp['deaths'])
        else:
            percent_revive_lst.append(temp['deaths'] / np.sum(down_total))

    t['percent_down_die'] = percent_revive_lst
    mu_downs = np.mean(down_total_lst)
    mu_not_percent_revived = np.mean(percent_revive_lst)

    if doc_filter.map_choice == 'mp_d':
        for circle in circle_lst:
            t[circle + '_percent'] = t[circle] / t['deaths']
    else:
        for circle in circle_lst:
            t[circle+'_percent'] = (t[circle] * t['percent_down_die']) / t['deaths']

    if doc_filter.map_choice == 'mp_d':
        final_dic = {'mu_playerCount': mu_playercount,
                     'mu_deaths_per_match': mu_downs}
        for circle in circle_lst:
            if '5' not in circle and '6' not in circle:
                final_dic[circle + '_kill_mean'] = (t[circle + '_percent'].mean()) * mu_playercount * 0.50
            else:
                final_dic[circle + '_kill_mean'] = (t[circle + '_percent'].mean()) * mu_playercount * 0.50

        for circle in circle_lst:
            if '5' not in circle and '6' not in circle:
                final_dic[circle + '_kill_std'] = t[circle + '_percent'].std(ddof=1) * mu_playercount
            else:
                final_dic[circle + '_kill_std'] = t[circle + '_percent'].std(ddof=1) * mu_playercount
    else:
        final_dic = {'mu_playerCount': mu_playercount,
                     'mu_downs_per_match': mu_downs,
                     'mu_downs_end_up_dead_percent': mu_not_percent_revived}
        for circle in circle_lst:
            final_dic[circle + '_kill_mean'] = (t[circle + '_percent'].mean()) * mu_playercount

        for circle in circle_lst:
            final_dic[circle + '_kill_std'] = t[circle + '_percent'].std(ddof=1) * mu_playercount
    return pd.DataFrame.from_dict(final_dic, orient='index', columns=['Deaths Per Circle']).round(1)


def engagement_mm(doc_filter: DocumentFilter) -> pd.DataFrame:
    data = doc_filter.df
    start_lst = list(data['startDateTime'])
    end_lst = list(data['endDateTime'])
    time_lst = list(data['timePlayed'])
    play_period = []
    placement = []
    for ind, val in enumerate(data['placementPercent']):
        place = data.iloc[ind]['placementPercent']
        temp_play_period = []
        for i_i, t_i in enumerate(end_lst[:ind]):
            if i_i > 1:
                previous_end_time = end_lst[i_i - 1]
                current_start_time = start_lst[i_i]
                current_playtime = time_lst[i_i]
                current_playtime_plus_buffer = datetime.timedelta(minutes=current_playtime + 900)
                if current_start_time <= (previous_end_time + current_playtime_plus_buffer):
                    temp_play_period.append(i_i)

        count = ind - 1
        temp_lst = []
        for i in temp_play_period[::-1]:
            if i == count:
                temp_lst.append(i)
                count -= 1
            else:
                break
        play_period.append(temp_lst[::-1])
        placement.append(place)

    mu_placement = np.mean(data['placementPercent'])
    x_lst = []
    y_lst = []
    period_lens = []
    for period in play_period:
        if len(period) > 5:
            period_lens.append(len(period))
            vals = data.iloc[period]
            x_i = np.sum(vals['timePlayed'])
            x_lst.append(x_i)
            y_i = (list(vals['placementPercent'])[-1] - mu_placement) / mu_placement
            # y_i = list(vals['placementPercent'])[-1]
            y_lst.append(y_i)

    final_df = pd.DataFrame()
    final_df['playtime_sum'] = x_lst
    final_df['final_placement'] = y_lst
    final_df['number_of_games'] = period_lens
    final_df['x_normalized'] = normalize(np.array(final_df['playtime_sum']))
    # x = normalize(np.array(final_df['playtime_sum']))

    from statsmodels.graphics.gofplots import qqplot
    qqplot(final_df['x_normalized'], line='s')
    plt.show()

    from scipy.stats import shapiro
    stat, p = shapiro(final_df['x_normalized'])
    print('Statistics=%.3f, p=%.3f' % (stat, p))

    import statsmodels.api as sm
    X = sm.add_constant(final_df['x_normalized'])
    y = final_df['final_placement']
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())

    y = final_df['final_placement']
    x = final_df['x_normalized']
    m, b = np.polyfit(x, y, 1)
    plt.scatter(y=y, x=x)
    plt.ylabel('Final Placement Percent')
    plt.xlabel('Play Time Sum')
    plt.title('Final Placement x Play Time Sum')
    plt.plot(x, m * x + b, color='tab:orange')
    plt.grid(linewidth=1, linestyle=(0, (5, 5)), alpha=.75)
    plt.show()
    return final_df


def hackers_overtime(doc_filter: DocumentFilter):
    data = doc_filter.df
    unique_match_ids = doc_filter.unique_ids
    lst = [np.mean(data[data['matchID'] == i]['hackerProb']) for i in unique_match_ids]
    lst_ind = [list(data[data['matchID'] == i]['startDateTime'])[0] for i in unique_match_ids]
    arr = np.array(lst)
    cm = cumulative_mean(arr)
    rn = running_mean(arr, 50)

    if data.iloc[0]['map'] == 'mp_d':
        title_label = 'Hackers Over Time: Verdansk'
    else:
        title_label = 'Hackers Over Time: Rebirth'

    ban = ['2021-07-16', '2021-03-23', '2021-04-07', '2021-02-02', '2021-05-14', '2021-08-11']
    ban_time = [datetime.datetime.strptime(i, '%Y-%m-%d') for i in ban]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(title_label, fontsize='xx-large')
    ax.plot(pd.DataFrame(cm, index=lst_ind), label="Cumulative Mean", color='tab:blue')
    ax.plot(pd.DataFrame(rn, index=lst_ind), label="Running Mean: Last 50 Games", color='tab:orange')

    ax.vlines(x=ban_time,
              ymin=np.min([np.min(rn), np.min(cm)]),
              ymax=np.max([np.max(cm), np.max(rn)]),
              color='red',
              linestyle=(0, (4, 4)),
              label='Activision Bans')

    ax.grid(linewidth=1, linestyle=(0, (5, 5)), alpha=.75)
    plt.legend(loc='lower right',
               fontsize='large',
               frameon=True)
    plt.show()
    return


def squad_effect(doc_filter: DocumentFilter, username: str, username_dic: dict):
    data = doc_filter.df
    col_lst = ['headshots', 'kills', 'deaths', 'kdRatio', 'scorePerMinute', 'distanceTraveled',
               'objectiveBrKioskBuy', 'percentTimeMoving', 'longestStreak', 'damageDone', 'damageTaken',
               'missionsComplete', 'objectiveLastStandKill', 'objectiveBrDownEnemyCircle1',
               'objectiveBrDownEnemyCircle2', 'objectiveBrDownEnemyCircle3', 'objectiveBrDownEnemyCircle4',
               'objectiveBrDownEnemyCircle5', 'objectiveBrDownEnemyCircle6', 'objectiveTeamWiped', 'objectiveReviver',
               'objectiveMunitionsBoxTeammateUsed', 'objectiveBrCacheOpen', 'objectiveMedalScoreKillSsRadarDrone',
               'placementPercent']
    squad_uno_lst = list(data['uno'].unique())
    user_uno = username_dic[username]
    name_dic = {}
    for uno in squad_uno_lst:
        temp_df = data[data['uno'] == uno].fillna(0.0)

        temp_dic = {}
        for col in col_lst:
            temp_dic[col] = np.mean(temp_df[col])

        name_dic[uno] = temp_dic

    name_df = pd.DataFrame.from_dict(name_dic, orient='index')
    lower_df = name_df[name_df['kdRatio'] < name_df.loc[user_uno]['kdRatio']]
    upper_df = name_df[name_df['kdRatio'] > name_df.loc[user_uno]['kdRatio']]

    user_placment = name_df.loc[user_uno]['placementPercent']
    user_placment_std = np.std(data[data['uno'] == user_uno]['placementPercent'], ddof=1)
    lower_placement_mu = lower_df['placementPercent'].mean()
    upper_placement_mu = upper_df['placementPercent'].mean()
    upper_change = (upper_placement_mu - user_placment) / user_placment
    lower_change = (lower_placement_mu - user_placment) / user_placment

    fig1, ax = plt.subplots(figsize=(10, 7))
    ax.set_title('Lower vs Higher Players', fontsize='xx-large')
    ax.boxplot([lower_df['placementPercent'], upper_df['placementPercent']])
    plt.axhline(user_placment, c='red', linestyle=(0, (4, 4)))
    plt.axhline(user_placment + user_placment_std, c='red', linestyle=(0, (4, 4)), alpha=0.25)
    plt.axhline(user_placment - user_placment_std, c='red', linestyle=(0, (4, 4)), alpha=0.25)
    ax.grid(linewidth=1, linestyle=(0, (5, 5)), alpha=0.75)
    plt.show()
    return
