"""Various one off plots.

Usage:
 ./warzone/plots.py

Author:
 Peter Rigali - 2021-08-30
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Optional
from warzone.classes.document_filter import DocumentFilter
from pyjr.plot.line import Line
from pyjr.classes.data import Data
from pyjr.classes.preprocess_data import PreProcess
from pyjr.utils.tools.math import _mean
from pyjr.utils.tools.clean import _prep


def personal_plot(doc_filter: DocumentFilter) -> None:
    """

    Returns a series of plots to visualize a users data.

    :param doc_filter: A DocumentFilter.
    :type doc_filter: DocumentFilter
    :return: *None*
    :example: *None*
    :note: This is intended to be used with map_choice, mode_choice, team_size and a Gamertag or uno inputted into the DocumentFilter.

    """
    data = doc_filter.df
    dates = list(data['startDate'].unique())
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(30, 30))

    if doc_filter.username is None:
        plt.title('Personal Data for: ' + doc_filter.uno, fontsize='xx-large')
    else:
        plt.title('Personal Data for: ' + doc_filter.username, fontsize='xx-large')

    # win/loss
    win_count_lst = []
    game_count_lst = []
    wl_ratio_lst = []
    for i in dates:
        temp = data[data['startDate'] == i]
        wins, losses = len(temp[temp['teamPlacement'] == 1]), len(temp[temp['teamPlacement'] > 1])
        win_count_lst.append(wins), game_count_lst.append(losses + wins), wl_ratio_lst.append(wins / (wins + losses))

    rm_wl = PreProcess(data=Data(data=wl_ratio_lst, name='wins', stats=False)).add_running(window=50).data
    cm_wl = PreProcess(data=Data(data=wl_ratio_lst, name='wins', stats=False)).add_cumulative().data
    ax[0, 0].set_title('Daily Win / Loss Ratio', fontsize='xx-large')
    ax[0, 0].plot(cm_wl, label='W/L Ratio Cumulative Mean', color='tab:blue')
    ax[0, 0].plot(rm_wl, label='W/L Ratio Running Mean', color='tab:blue', alpha=0.25)
    ax[0, 0].legend(loc='lower left', fontsize='large', frameon=True, framealpha=0.85)
    ax2 = ax[0, 0].twinx()
    ax2.plot(Data(data=game_count_lst, name='losses', stats=False).data, label='Daily Game Count', color='black', alpha=0.25)
    ax2.legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)
    # ax2.set_xticks(np.arange(min(range(len(wl_df))), max(range(len(wl_df))) + 1, 100.0))

    # placement
    rm_wl = PreProcess(data=Data(data=data['placementPercent'], name='placement', stats=False)).add_running(window=50).data
    cm_wl = PreProcess(data=Data(data=data['placementPercent'], name='placement', stats=False)).add_cumulative().data
    ax[0, 1].set_title('Team Placement', fontsize='xx-large')
    ax[0, 1].plot(cm_wl, label='Placement Cumulative Mean', color='tab:blue')
    ax[0, 1].plot(rm_wl, label='Placement Running Mean', color='tab:blue', alpha=0.25)
    ax[0, 1].set_xticks(np.arange(min(range(len(data['matchID']))), max(range(len(data['matchID']))) + 1, 100.0))
    ax[0, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # kd
    rm_wl = PreProcess(data=Data(data=data['kdRatio'], name='kd', stats=False)).add_running(window=50).data
    cm_wl = PreProcess(data=Data(data=data['kdRatio'], name='kd', stats=False)).add_cumulative().data
    ax[1, 0].set_title('Kill Death Ratio', fontsize='xx-large')
    ax[1, 0].plot(cm_wl, label='Kd Ratio Cumulative Mean', color='tab:blue')
    ax[1, 0].plot(rm_wl, label='Kd Ratio Running Mean', color='tab:blue', alpha=0.25)
    ax[1, 0].set_xticks(np.arange(min(range(len(data['matchID']))), max(range(len(data['matchID']))) + 1, 100.0))
    ax[1, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Kills and Deaths
    ax[1, 1].set_title('Kills and Deaths Per Game', fontsize='xx-large')
    rm_wl = PreProcess(data=Data(data=data['kills'], name='kills', stats=False)).add_running(window=50).data
    cm_wl = PreProcess(data=Data(data=data['kills'], name='kills', stats=False)).add_cumulative().data
    rm_wl1 = PreProcess(data=Data(data=data['deaths'], name='deaths', stats=False)).add_running(window=50).data
    cm_wl1 = PreProcess(data=Data(data=data['deaths'], name='deaths', stats=False)).add_cumulative().data
    ax[1, 1].set_title('Kills and Deaths', fontsize='xx-large')
    ax[1, 1].plot(cm_wl, label='Kills Cumulative Mean', color='green')
    ax[1, 1].plot(cm_wl1, label='Deaths Cumulative Mean', color='red')
    ax[1, 1].plot(rm_wl, label='Kills Running Mean', color='green', alpha=0.25)
    ax[1, 1].plot(rm_wl1, label='Deaths Running Mean', color='red', alpha=0.25)
    ax[1, 1].set_xticks(np.arange(min(range(len(data['matchID']))), max(range(len(data['matchID']))) + 1, 100.0))
    ax[1, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Damage
    rm_wl = PreProcess(data=Data(data=data['damageDone'], name='kills', stats=False)).add_running(window=50).data
    cm_wl = PreProcess(data=Data(data=data['damageDone'], name='kills', stats=False)).add_cumulative().data
    rm_wl1 = PreProcess(data=Data(data=data['damageTaken'], name='deaths', stats=False)).add_running(window=50).data
    cm_wl1 = PreProcess(data=Data(data=data['damageTaken'], name='deaths', stats=False)).add_cumulative().data
    ax[2, 0].set_title('Damage', fontsize='xx-large')
    ax[2, 0].plot(cm_wl, label='Damage Done Cumulative Mean', color='green')
    ax[2, 0].plot(cm_wl1, label='Damage Taken Cumulative Mean', color='red')
    ax[2, 0].plot(rm_wl, label='Damage Done Running Mean', color='green', alpha=0.25)
    ax[2, 0].plot(rm_wl1, label='Damage Taken Running Mean', color='red', alpha=0.25)
    ax[2, 0].set_xticks(np.arange(min(range(len(data['matchID']))), max(range(len(data['matchID']))) + 1, 100.0))
    ax[2, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Misc
    ax[2, 1].set_title('Misc', fontsize='xx-large')
    ax[2, 1].plot(data['headshots'])
    ax[2, 1].set_xticks(np.arange(min(range(len(data['matchID']))), max(range(len(data['matchID']))) + 1, 100.0))
    ax[2, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    plt.show()


def lobby_plot(doc_filter: DocumentFilter) -> None:
    """

    Returns a series of plots.

    :param doc_filter: A DocumentFilter.
    :type doc_filter: DocumentFilter
    :return: *None*
    :example: *None*
    :note: This is intended to be used with map_choice and mode_choice inputted into the DocumentFilter.

    """
    data = doc_filter.df
    games = doc_filter.unique_match_ids
    dates = tuple(data['startDate'].unique())
    col_lst = ('kdRatio', 'kills', 'deaths', 'damageDone', 'damageTaken', 'percentTimeMoving', 'distanceTraveled',
               'objectiveTeamWiped', 'objectiveReviver', 'missionsComplete')

    day_dic = {}
    for date in dates:
        temp_df = data[data['startDate'] == date].fillna(0.0)
        day_dic[date] = [_mean(_prep(d=temp_df[col])) for col in col_lst]

    game_dic = {}
    for game in games:
        temp_df = data[data['matchID'] == game].fillna(0)
        # game_dic[game] = [np.mean(temp_df[col]) for col in col_lst]
        game_dic[game] = [_mean(_prep(d=temp_df[col])) for col in col_lst]

    day_df = pd.DataFrame.from_dict(day_dic, orient='index', columns=col_lst).fillna(0)
    game_df = pd.DataFrame.from_dict(game_dic, orient='index', columns=col_lst).fillna(0)

    fig, ax = plt.subplots(nrows=8, ncols=2, figsize=(30, 50))
    plt.title('Lobby Data for', fontsize='xx-large')

    # kd
    c = PreProcess(data=Data(data=day_df['kdRatio'], stats=False)).add_cumulative().data
    r = PreProcess(data=Data(data=day_df['kdRatio'], stats=False)).add_running(window=50).data
    ax[0, 0].set_title('Kill Death Ratio Per Day', fontsize='xx-large')
    ax[0, 0].plot(c, label='Kd Ratio Cumulative Mean', color='tab:blue')
    ax[0, 0].plot(r, label='Kd Ratio Running Mean', color='tab:blue', alpha=0.25)
    # ax[0, 0].set_xticks(np.arange(min(range(len(day_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[0, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    c = PreProcess(data=Data(data=game_df['kdRatio'], stats=False)).add_cumulative().data
    r = PreProcess(data=Data(data=game_df['kdRatio'], stats=False)).add_running(window=50).data
    ax[0, 1].set_title('Kill Death Ratio Per Game', fontsize='xx-large')
    ax[0, 1].plot(c, label='Kd Ratio Cumulative Mean', color='tab:blue')
    ax[0, 1].plot(r, label='Kd Ratio Running Mean', color='tab:blue', alpha=0.25)
    # ax[0, 1].set_xticks(np.arange(min(range(len(day_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[0, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Kills and Deaths
    ax[1, 0].set_title('Kills and Deaths Per Game', fontsize='xx-large')
    c = PreProcess(data=Data(data=day_df['kills'], stats=False)).add_cumulative().data
    r = PreProcess(data=Data(data=day_df['kills'], stats=False)).add_running(window=50).data
    c1 = PreProcess(data=Data(data=day_df['deaths'], stats=False)).add_cumulative().data
    r1 = PreProcess(data=Data(data=day_df['deaths'], stats=False)).add_running(window=50).data
    ax[1, 0].set_title('Kills and Deaths Per Day', fontsize='xx-large')
    ax[1, 0].plot(c, label='Kills Cumulative Mean', color='green')
    ax[1, 0].plot(c1, label='Deaths Cumulative Mean', color='red')
    ax[1, 0].plot(r, label='Kills Running Mean', color='green', alpha=0.25)
    ax[1, 0].plot(r1, label='Deaths Running Mean', color='red', alpha=0.25)
    # ax[1, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[1, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    ax[1, 1].set_title('Kills and Deaths Per Game', fontsize='xx-large')
    c = PreProcess(data=Data(data=game_df['kills'], stats=False)).add_cumulative().data
    r = PreProcess(data=Data(data=game_df['kills'], stats=False)).add_running(window=50).data
    c1 = PreProcess(data=Data(data=game_df['deaths'], stats=False)).add_cumulative().data
    r1 = PreProcess(data=Data(data=game_df['deaths'], stats=False)).add_running(window=50).data
    ax[1, 1].set_title('Kills and Deaths Per Game', fontsize='xx-large')
    ax[1, 1].plot(c, label='Kills Cumulative Mean', color='green')
    ax[1, 1].plot(c1, label='Deaths Cumulative Mean', color='red')
    ax[1, 1].plot(r, label='Kills Running Mean', color='green', alpha=0.25)
    ax[1, 1].plot(r1, label='Deaths Running Mean', color='red', alpha=0.25)
    # ax[1, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[1, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Damage
    c = PreProcess(data=Data(data=day_df['damageDone'], stats=False)).add_cumulative().data
    r = PreProcess(data=Data(data=day_df['damageTaken'], stats=False)).add_running(window=50).data
    c1 = PreProcess(data=Data(data=day_df['damageDone'], stats=False)).add_cumulative().data
    r1 = PreProcess(data=Data(data=day_df['damageTaken'], stats=False)).add_running(window=50).data
    ax[2, 0].set_title('Damage Per Day', fontsize='xx-large')
    ax[2, 0].plot(c, label='Damage Done Cumulative Mean', color='green')
    ax[2, 0].plot(c1, label='Damage Taken Cumulative Mean', color='red')
    ax[2, 0].plot(r, label='Damage Done Running Mean', color='green', alpha=0.25)
    ax[2, 0].plot(r1, label='Damage Taken Running Mean', color='red', alpha=0.25)
    # ax[2, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[2, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    c = PreProcess(data=Data(data=game_df['damageDone'], stats=False)).add_cumulative().data
    r = PreProcess(data=Data(data=game_df['damageTaken'], stats=False)).add_running(window=50).data
    c1 = PreProcess(data=Data(data=game_df['damageDone'], stats=False)).add_cumulative().data
    r1 = PreProcess(data=Data(data=game_df['damageTaken'], stats=False)).add_running(window=50).data
    ax[2, 1].set_title('Damage Per Game', fontsize='xx-large')
    ax[2, 1].plot(c, label='Damage Done Cumulative Mean', color='green')
    ax[2, 1].plot(c1, label='Damage Taken Cumulative Mean', color='red')
    ax[2, 1].plot(r, label='Damage Done Running Mean', color='green', alpha=0.25)
    ax[2, 1].plot(r1, label='Damage Taken Running Mean', color='red', alpha=0.25)
    # ax[2, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[2, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Time Moving
    c = PreProcess(data=Data(data=day_df['percentTimeMoving'], stats=False)).add_cumulative().data
    r = PreProcess(data=Data(data=day_df['percentTimeMoving'], stats=False)).add_running(window=50).data
    ax[3, 0].set_title('Time Moving Per Day', fontsize='xx-large')
    ax[3, 0].plot(c, label='Time Moving Cumulative Mean', color='tab:blue')
    ax[3, 0].plot(r, label='Time Moving Running Mean', color='tab:blue', alpha=0.25)
    # ax[3, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[3, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    c = PreProcess(data=Data(data=game_df['percentTimeMoving'], stats=False)).add_cumulative().data
    r = PreProcess(data=Data(data=game_df['percentTimeMoving'], stats=False)).add_running(window=50).data
    ax[3, 1].set_title('Time Moving Per Game', fontsize='xx-large')
    ax[3, 1].plot(c, label='Time Moving Cumulative Mean', color='tab:blue')
    ax[3, 1].plot(r, label='Time Moving Running Mean', color='tab:blue', alpha=0.25)
    # ax[3, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[3, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Distance Traveled
    c = PreProcess(data=Data(data=day_df['distanceTraveled'], stats=False)).add_cumulative().data
    r = PreProcess(data=Data(data=day_df['distanceTraveled'], stats=False)).add_running(window=50).data
    ax[4, 0].set_title('Distance Traveled Per Day', fontsize='xx-large')
    ax[4, 0].plot(c, label='Distance Traveled Cumulative Mean', color='tab:blue')
    ax[4, 0].plot(r, label='Distance Traveled Running Mean', color='tab:blue', alpha=0.25)
    # ax[4, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[4, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    c = PreProcess(data=Data(data=game_df['distanceTraveled'], stats=False)).add_cumulative().data
    r = PreProcess(data=Data(data=game_df['distanceTraveled'], stats=False)).add_running(window=50).data
    ax[4, 1].set_title('Distance Traveled Per Game', fontsize='xx-large')
    ax[4, 1].plot(c, label='Distance Traveled Cumulative Mean', color='tab:blue')
    ax[4, 1].plot(r, label='Distance Traveled Running Mean', color='tab:blue', alpha=0.25)
    # ax[4, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[4, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Team Wipes
    c = PreProcess(data=Data(data=day_df['objectiveTeamWiped'], stats=False)).add_cumulative().data
    r = PreProcess(data=Data(data=day_df['objectiveTeamWiped'], stats=False)).add_running(window=50).data
    ax[5, 0].set_title('Team Wipes Per Day', fontsize='xx-large')
    ax[5, 0].plot(c, label='Team Wipes Cumulative Mean', color='tab:blue')
    ax[5, 0].plot(r, label='Team Wipes Running Mean', color='tab:blue', alpha=0.25)
    # ax[5, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[5, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    c = PreProcess(data=Data(data=game_df['objectiveTeamWiped'], stats=False)).add_cumulative().data
    r = PreProcess(data=Data(data=game_df['objectiveTeamWiped'], stats=False)).add_running(window=50).data
    ax[5, 1].set_title('Team Wipes Per Game', fontsize='xx-large')
    ax[5, 1].plot(c, label='Team Wipes Cumulative Mean', color='tab:blue')
    ax[5, 1].plot(r, label='Team Wipes Running Mean', color='tab:blue', alpha=0.25)
    # ax[5, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[5, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Revives
    c = PreProcess(data=Data(data=day_df['objectiveReviver'], stats=False)).add_cumulative().data
    r = PreProcess(data=Data(data=day_df['objectiveReviver'], stats=False)).add_running(window=50).data
    ax[6, 0].set_title('Revives Per Day', fontsize='xx-large')
    ax[6, 0].plot(c, label='Revives Cumulative Mean', color='tab:blue')
    ax[6, 0].plot(r, label='Revives Running Mean', color='tab:blue', alpha=0.25)
    # ax[6, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[6, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    c = PreProcess(data=Data(data=game_df['objectiveReviver'], stats=False)).add_cumulative().data
    r = PreProcess(data=Data(data=game_df['objectiveReviver'], stats=False)).add_running(window=50).data
    ax[6, 1].set_title('Revives Per Game', fontsize='xx-large')
    ax[6, 1].plot(c, label='Revives Cumulative Mean', color='tab:blue')
    ax[6, 1].plot(r, label='Revives Running Mean', color='tab:blue', alpha=0.25)
    # ax[6, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[6, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Missions Complete
    c = PreProcess(data=Data(data=day_df['missionsComplete'], stats=False)).add_cumulative().data
    r = PreProcess(data=Data(data=day_df['missionsComplete'], stats=False)).add_running(window=50).data
    ax[7, 0].set_title('Missions Complete Per Day', fontsize='xx-large')
    ax[7, 0].plot(c, label='Missions Complete Cumulative Mean', color='tab:blue')
    ax[7, 0].plot(r, label='Missions Complete Running Mean', color='tab:blue', alpha=0.25)
    # ax[7, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[7, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    c = PreProcess(data=Data(data=game_df['missionsComplete'], stats=False)).add_cumulative().data
    r = PreProcess(data=Data(data=game_df['missionsComplete'], stats=False)).add_running(window=50).data
    ax[7, 1].set_title('Missions Complete Per Game', fontsize='xx-large')
    ax[7, 1].plot(c, label='Missions Complete Cumulative Mean', color='tab:blue')
    ax[7, 1].plot(r, label='Missions Complete Running Mean', color='tab:blue', alpha=0.25)
    # ax[5, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[7, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)
    plt.show()


def squad_plot(doc_filter: DocumentFilter, col_lst: Optional[List[str]] = None) -> None:
    """

    Build a Polar plot for visualizing squad stats.

    :param doc_filter: A DocumentFilter.
    :type doc_filter: DocumentFilter
    :param col_lst: Input List of Columns to analyze.
    :type col_lst: List[str] or str
    :return: *None*
    :example: *None*
    :note: This is intended to be used with map_choice and mode_choice inputted into the DocumentFilter.

    """
    data = doc_filter.df
    if col_lst is None:
        col_lst = ['kdRatio', 'kills', 'deaths', 'damageDone', 'damageTaken', 'objectiveTeamWiped', 'objectiveReviver',
                   'missionsComplete']

    people_dic = {}
    for i in doc_filter.username:
        temp_df = data[data['uno'] == doc_filter.username_dic[i]]
        people_dic[i] = {j: _mean(_prep(d=temp_df[j])) for j in col_lst}

    people_df = pd.DataFrame.from_dict(people_dic, orient='index')
    normalized_df = (people_df - people_df.loc[doc_filter.username[0]]) / people_df.loc[doc_filter.username[0]] + 1

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, polar=True)

    n = len(doc_filter.username) - 1
    cmap = [plt.get_cmap('viridis')(1. * i / n) for i in range(n)]
    theta = np.linspace(0, 2 * np.pi, len(col_lst) + 1)
    count = 0
    for ind1, person in enumerate(doc_filter.username):
        row = list(normalized_df.loc[person])
        if person == 'Claim':
            ax.plot(theta, row + [row[0]], color='tab:orange', linewidth=4, alpha=1, linestyle=(0, (4, 2)))
        else:
            ax.plot(theta, row + [row[0]], color=cmap[count], linewidth=2, alpha=0.50)
            count += 1

    ax.legend(labels=doc_filter.username,
              loc='upper left',
              fontsize='large',
              frameon=True,
              bbox_to_anchor=(1.05, 1)).get_frame().set_linewidth(2)

    col_lst_n = []
    for i in col_lst:
        col_lst_n.append(i)
        col_lst_n.append('')

    # Highlight user gridline
    # gridlines = ax.yaxis.get_gridlines()
    # temp_lines = [i / len(gridlines) for i in range(1, len(gridlines))] + [1]
    # ax.set_yticklabels(temp_lines)
    # ind = temp_lines.index(0.625)
    # gridlines[ind].set_color("black")
    # gridlines[ind].set_linestyle((0, (5, 10)))
    # gridlines[ind].set_linewidth(2)

    ax.xaxis.set_ticks(theta)
    ax.xaxis.set_ticklabels(col_lst + [''], fontsize='large')
    ax.grid(linewidth=1, linestyle=(0, (5, 5)), alpha=.75)

    # Hide x ticks
    # ax.tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=False)
    plt.show()


# def find_high_low_points(data, distance: int = 50):
#     from warzone.utils.base import normalize
#     from scipy.signal import chirp, find_peaks, peak_widths
#     datan = normalize(arr=data)
#
#     # Find best params
#     peaks, _ = find_peaks(x=datan, height=np.quantile(datan, .841), distance=distance, prominence=(None, None),
#                           width=(10, None), threshold=(None, None), wlen=None, rel_height=1.0,
#                           plateau_size=(1, None))
#     valleys, _ = find_peaks(x=datan * -1, height=-1 * np.quantile(datan, .159), distance=distance,
#                             prominence=(None, None), width=(10, None), threshold=(None, None), wlen=None,
#                             rel_height=1.0, plateau_size=(1, None))
#     mu1 = int(np.mean([j - peaks[i - 1] for i, j in enumerate(peaks) if i != 0]))
#     mu2 = int(np.mean([j - valleys[i - 1] for i, j in enumerate(valleys) if i != 0]))
#
#     # Fit with params
#     peaks, peak_prop = find_peaks(x=datan, height=np.quantile(datan, .841), distance=distance,
#                                   prominence=(None, None), width=(10, None), threshold=(None, None), wlen=mu1,
#                                   rel_height=1.0, plateau_size=(1, None))
#     valleys, valley_prop = find_peaks(x=datan * -1, height=-1 * np.quantile(datan, .159), distance=distance,
#                                       prominence=(None, None), width=(10, None), threshold=(None, None), wlen=mu2,
#                                       rel_height=1.0, plateau_size=(1, None))
#
#     # Plot
#     plt.scatter(peaks, datan[peaks], color='red')
#     plt.vlines(x=peaks, ymin=datan[peaks] - peak_prop['prominences'], ymax=datan[peaks], color="red")
#     plt.hlines(y=peak_prop["width_heights"], xmin=peak_prop["left_ips"], xmax=peak_prop["right_ips"], color="red")
#
#     plt.scatter(valleys, datan[valleys], color='green')
#     plt.vlines(x=valleys, ymin=datan[valleys] + valley_prop["prominences"], ymax=datan[valleys], color="green")
#     plt.hlines(y=valley_prop["width_heights"] * -1, xmin=valley_prop["left_ips"], xmax=valley_prop["right_ips"],
#                color="green")
#
#     plt.plot(datan, color='grey', alpha=.75)
#     plt.show()
#
#     # Ratios as descriptive
#     mu = np.mean(datan)
#     peak_heights = np.sum(datan[peaks] - peak_prop['prominences'])
#     valley_heights = np.sum(datan[valleys] + valley_prop['prominences'])
#     height_ratio = peak_heights / (peak_heights + valley_heights)
#
#     peak_widths = np.sum(peak_prop["right_ips"] - peak_prop["left_ips"])
#     valley_widths =  np.sum(valley_prop["right_ips"] - valley_prop["left_ips"])
#     width_ratio = peak_widths / (peak_widths + valley_widths)
#
#     # Slopes as descriptive
#     con = np.concatenate([peaks, valleys])
#     con.sort()
#
#     mu = np.mean(datan)
#     count = 0
#     for i in datan:
#         if i < mu:
#             count += 1
#         else:
#             break
#
#     lst = [count] + list(con) + [list(datan.index)[-1]]
#     slopes = []
#     for i, j in enumerate(lst):
#         if i == 0:
#             continue
#         else:
#             x1 = lst[i - 1]
#             x2 = j
#             y1 = datan[lst[i - 1]]
#             y2 = datan[j]
#             m = (y2 - y1) / (x2 - x1)
#             slopes.append(m)
#     mu_slope = np.median(slopes)
#     return (peaks, peak_prop), (valleys, valley_prop)
#
# our_doc = DocumentFilter(input_df=cod.our_df, map_choice='rebirth', mode_choice='resurgence', team_size='quad', uno=cod.my_uno)
# ma50 = our_doc.df['kills'].rolling(window=50).mean().dropna().reset_index(drop=True)
# t = find_high_low_points(data=ma50)


def dist_plot(our_filter: DocumentFilter, other_filter: DocumentFilter,  limit: tuple,
              column: Optional[str] = 'kills') -> pd.DataFrame:
    od, wd = Data(data=our_filter.df[column], stats=False, unique=True), Data(data=other_filter.df[column], stats=False, unique=True)
    od, wd = {i: od.data.count(i) / od.len for i in od.unique}, {i: wd.data.count(i) / wd.len for i in wd.unique}

    for i in wd.keys():
        if i not in od.keys():
            od[i] = 0
    for i in od.keys():
        if i not in wd.keys():
            wd[i] = 0
    data = pd.DataFrame({'our': od, 'other': wd})
    Line(data=data, title='Distribution: ' + column, show=True, limit=limit)
    return data
