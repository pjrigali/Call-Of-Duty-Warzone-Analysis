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
from warzone.utils.base import running_mean, cumulative_mean
from warzone.classes.document_filter import DocumentFilter


def personal_plot(doc_filter: DocumentFilter) -> None:
    """

    Returns a series of plots.

    :param doc_filter: A DocumentFilter.
    :type doc_filter: DocumentFilter
    :return: *None*
    :example: *None*
    :note: This is intended to be used with map_choice, mode_choice and a Gamertag inputted into the DocumentFilter.

    """
    data = doc_filter.df
    dates = list(data['startDate'].unique())
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(30, 30))
    plt.title('Personal Data for: ' + doc_filter.username, fontsize='xx-large')

    # win/loss
    win_count_lst = []
    game_count_lst = []
    wl_ratio_lst = []
    for i in dates:
        temp = data[data['startDate'] == i]
        wins = len(temp[temp['teamPlacement'] == 1])
        losses = len(temp[temp['teamPlacement'] > 1])
        win_count_lst.append(wins)
        game_count_lst.append(losses + wins)
        wl_ratio_lst.append(wins / (wins + losses))

    wl_df = pd.DataFrame(wl_ratio_lst, columns=['ratio'], index=dates)
    wl_df['wins'] = win_count_lst
    wl_df['losses'] = game_count_lst
    cm_wl = cumulative_mean(np.array(wl_df['ratio']))
    rm_wl = running_mean(np.array(wl_df['ratio']), 50)
    ax[0, 0].set_title('Daily Win / Loss Ratio', fontsize='xx-large')
    ax[0, 0].plot(cm_wl, label='W/L Ratio Cumulative Mean', color='tab:blue')
    ax[0, 0].plot(rm_wl, label='W/L Ratio Running Mean', color='tab:blue', alpha=0.25)
    ax[0, 0].legend(loc='lower left', fontsize='large', frameon=True, framealpha=0.85)
    ax2 = ax[0, 0].twinx()
    ax2.plot(np.array(wl_df['losses']), label='Daily Game Count', color='black', alpha=0.25)
    ax2.legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)
    # ax2.set_xticks(np.arange(min(range(len(wl_df))), max(range(len(wl_df))) + 1, 100.0))

    # placement
    cm_p = cumulative_mean(np.array(data['placementPercent']))
    rm_p = running_mean(np.array(data['placementPercent']), 50)
    ax[0, 1].set_title('Team Placement', fontsize='xx-large')
    ax[0, 1].plot(cm_p, label='Placement Cumulative Mean', color='tab:blue')
    ax[0, 1].plot(rm_p, label='Placement Running Mean', color='tab:blue', alpha=0.25)
    ax[0, 1].set_xticks(np.arange(min(range(len(data['matchID']))), max(range(len(data['matchID']))) + 1, 100.0))
    ax[0, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # kd
    cm_kd = cumulative_mean(np.array(data['kdRatio']))
    rm_kd = running_mean(np.array(data['kdRatio']), 50)
    ax[1, 0].set_title('Kill Death Ratio', fontsize='xx-large')
    ax[1, 0].plot(cm_kd, label='Kd Ratio Cumulative Mean', color='tab:blue')
    ax[1, 0].plot(rm_kd, label='Kd Ratio Running Mean', color='tab:blue', alpha=0.25)
    ax[1, 0].set_xticks(np.arange(min(range(len(data['matchID']))), max(range(len(data['matchID']))) + 1, 100.0))
    ax[1, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Kills and Deaths
    ax[1, 1].set_title('Kills and Deaths Per Game', fontsize='xx-large')
    cm_kills = cumulative_mean(np.array(data['kills']))
    cm_deaths = cumulative_mean(np.array(data['deaths']))
    rm_kills = running_mean(np.array(data['kills']), 50)
    rm_deaths = running_mean(np.array(data['deaths']), 50)
    ax[1, 1].set_title('Kills and Deaths', fontsize='xx-large')
    ax[1, 1].plot(cm_kills, label='Kills Cumulative Mean', color='green')
    ax[1, 1].plot(cm_deaths, label='Deaths Cumulative Mean', color='red')
    ax[1, 1].plot(rm_kills, label='Kills Running Mean', color='green', alpha=0.25)
    ax[1, 1].plot(rm_deaths, label='Deaths Running Mean', color='red', alpha=0.25)
    ax[1, 1].set_xticks(np.arange(min(range(len(data['matchID']))), max(range(len(data['matchID']))) + 1, 100.0))
    ax[1, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Damage
    cm_dam_d = cumulative_mean(np.array(data['damageDone']))
    cm_dam_t = cumulative_mean(np.array(data['damageTaken']))
    rm_dam_d = running_mean(np.array(data['damageDone']), 50)
    rm_dam_t = running_mean(np.array(data['damageTaken']), 50)
    ax[2, 0].set_title('Damage', fontsize='xx-large')
    ax[2, 0].plot(cm_dam_d, label='Damage Done Cumulative Mean', color='green')
    ax[2, 0].plot(cm_dam_t, label='Damage Taken Cumulative Mean', color='red')
    ax[2, 0].plot(rm_dam_d, label='Damage Done Running Mean', color='green', alpha=0.25)
    ax[2, 0].plot(rm_dam_t, label='Damage Taken Running Mean', color='red', alpha=0.25)
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
    games = doc_filter.unique_ids
    dates = list(data['startDate'].unique())
    col_lst = ['kdRatio', 'kills', 'deaths', 'damageDone', 'damageTaken', 'percentTimeMoving', 'distanceTraveled',
               'objectiveTeamWiped', 'objectiveReviver', 'missionsComplete']

    day_dic = {}
    for date in dates:
        temp_df = data[data['startDate'] == date].fillna(0)
        day_dic[date] = [np.mean(temp_df[col]) for col in col_lst]

    game_dic = {}
    for game in games:
        temp_df = data[data['matchID'] == game].fillna(0)
        game_dic[game] = [np.mean(temp_df[col]) for col in col_lst]

    day_df = pd.DataFrame.from_dict(day_dic, orient='index', columns=col_lst).fillna(0)
    game_df = pd.DataFrame.from_dict(game_dic, orient='index', columns=col_lst).fillna(0)

    fig, ax = plt.subplots(nrows=8, ncols=2, figsize=(30, 50))
    plt.title('Lobby Data for', fontsize='xx-large')

    # kd
    cm_kd = cumulative_mean(np.array(day_df['kdRatio']))
    rm_kd = running_mean(np.array(day_df['kdRatio']), 50)
    ax[0, 0].set_title('Kill Death Ratio Per Day', fontsize='xx-large')
    ax[0, 0].plot(cm_kd, label='Kd Ratio Cumulative Mean', color='tab:blue')
    ax[0, 0].plot(rm_kd, label='Kd Ratio Running Mean', color='tab:blue', alpha=0.25)
    # ax[1, 0].set_xticks(np.arange(min(range(len(day_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[0, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    cm_kd = cumulative_mean(np.array(game_df['kdRatio']))
    rm_kd = running_mean(np.array(game_df['kdRatio']), 50)
    ax[0, 1].set_title('Kill Death Ratio Per Game', fontsize='xx-large')
    ax[0, 1].plot(cm_kd, label='Kd Ratio Cumulative Mean', color='tab:blue')
    ax[0, 1].plot(rm_kd, label='Kd Ratio Running Mean', color='tab:blue', alpha=0.25)
    # ax[0, 1].set_xticks(np.arange(min(range(len(day_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[0, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Kills and Deaths
    ax[1, 0].set_title('Kills and Deaths Per Game', fontsize='xx-large')
    cm_kills = cumulative_mean(np.array(day_df['kills']))
    cm_deaths = cumulative_mean(np.array(day_df['deaths']))
    rm_kills = running_mean(np.array(day_df['kills']), 50)
    rm_deaths = running_mean(np.array(day_df['deaths']), 50)
    ax[1, 0].set_title('Kills and Deaths Per Day', fontsize='xx-large')
    ax[1, 0].plot(cm_kills, label='Kills Cumulative Mean', color='green')
    ax[1, 0].plot(cm_deaths, label='Deaths Cumulative Mean', color='red')
    ax[1, 0].plot(rm_kills, label='Kills Running Mean', color='green', alpha=0.25)
    ax[1, 0].plot(rm_deaths, label='Deaths Running Mean', color='red', alpha=0.25)
    # ax[1, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[1, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    ax[1, 1].set_title('Kills and Deaths Per Game', fontsize='xx-large')
    cm_kills = cumulative_mean(np.array(game_df['kills']))
    cm_deaths = cumulative_mean(np.array(game_df['deaths']))
    rm_kills = running_mean(np.array(game_df['kills']), 50)
    rm_deaths = running_mean(np.array(game_df['deaths']), 50)
    ax[1, 1].set_title('Kills and Deaths Per Game', fontsize='xx-large')
    ax[1, 1].plot(cm_kills, label='Kills Cumulative Mean', color='green')
    ax[1, 1].plot(cm_deaths, label='Deaths Cumulative Mean', color='red')
    ax[1, 1].plot(rm_kills, label='Kills Running Mean', color='green', alpha=0.25)
    ax[1, 1].plot(rm_deaths, label='Deaths Running Mean', color='red', alpha=0.25)
    # ax[1, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[1, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Damage
    cm_dam_d = cumulative_mean(np.array(day_df['damageDone']))
    cm_dam_t = cumulative_mean(np.array(day_df['damageTaken']))
    rm_dam_d = running_mean(np.array(day_df['damageDone']), 50)
    rm_dam_t = running_mean(np.array(day_df['damageTaken']), 50)
    ax[2, 0].set_title('Damage Per Day', fontsize='xx-large')
    ax[2, 0].plot(cm_dam_d, label='Damage Done Cumulative Mean', color='green')
    ax[2, 0].plot(cm_dam_t, label='Damage Taken Cumulative Mean', color='red')
    ax[2, 0].plot(rm_dam_d, label='Damage Done Running Mean', color='green', alpha=0.25)
    ax[2, 0].plot(rm_dam_t, label='Damage Taken Running Mean', color='red', alpha=0.25)
    # ax[2, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[2, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    cm_dam_d = cumulative_mean(np.array(game_df['damageDone']))
    cm_dam_t = cumulative_mean(np.array(game_df['damageTaken']))
    rm_dam_d = running_mean(np.array(game_df['damageDone']), 50)
    rm_dam_t = running_mean(np.array(game_df['damageTaken']), 50)
    ax[2, 1].set_title('Damage Per Game', fontsize='xx-large')
    ax[2, 1].plot(cm_dam_d, label='Damage Done Cumulative Mean', color='green')
    ax[2, 1].plot(cm_dam_t, label='Damage Taken Cumulative Mean', color='red')
    ax[2, 1].plot(rm_dam_d, label='Damage Done Running Mean', color='green', alpha=0.25)
    ax[2, 1].plot(rm_dam_t, label='Damage Taken Running Mean', color='red', alpha=0.25)
    # ax[2, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[2, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Time Moving
    cm_kd = cumulative_mean(np.array(day_df['percentTimeMoving']))
    rm_kd = running_mean(np.array(day_df['percentTimeMoving']), 50)
    ax[3, 0].set_title('Time Moving Per Day', fontsize='xx-large')
    ax[3, 0].plot(cm_kd, label='Time Moving Cumulative Mean', color='tab:blue')
    ax[3, 0].plot(rm_kd, label='Time Moving Running Mean', color='tab:blue', alpha=0.25)
    # ax[3, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[3, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    cm_kd = cumulative_mean(np.array(game_df['percentTimeMoving']))
    rm_kd = running_mean(np.array(game_df['percentTimeMoving']), 50)
    ax[3, 1].set_title('Time Moving Per Game', fontsize='xx-large')
    ax[3, 1].plot(cm_kd, label='Time Moving Cumulative Mean', color='tab:blue')
    ax[3, 1].plot(rm_kd, label='Time Moving Running Mean', color='tab:blue', alpha=0.25)
    # ax[3, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[3, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Distance Traveled
    cm_kd = cumulative_mean(np.array(day_df['distanceTraveled']))
    rm_kd = running_mean(np.array(day_df['distanceTraveled']), 50)
    ax[4, 0].set_title('Distance Traveled Per Day', fontsize='xx-large')
    ax[4, 0].plot(cm_kd, label='Distance Traveled Cumulative Mean', color='tab:blue')
    ax[4, 0].plot(rm_kd, label='Distance Traveled Running Mean', color='tab:blue', alpha=0.25)
    # ax[4, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[4, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    cm_kd = cumulative_mean(np.array(game_df['distanceTraveled']))
    rm_kd = running_mean(np.array(game_df['distanceTraveled']), 50)
    ax[4, 1].set_title('Distance Traveled Per Game', fontsize='xx-large')
    ax[4, 1].plot(cm_kd, label='Distance Traveled Cumulative Mean', color='tab:blue')
    ax[4, 1].plot(rm_kd, label='Distance Traveled Running Mean', color='tab:blue', alpha=0.25)
    # ax[4, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[4, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Team Wipes
    cm_kd = cumulative_mean(np.array(day_df['objectiveTeamWiped']))
    rm_kd = running_mean(np.array(day_df['objectiveTeamWiped']), 50)
    ax[5, 0].set_title('Team Wipes Per Day', fontsize='xx-large')
    ax[5, 0].plot(cm_kd, label='Team Wipes Cumulative Mean', color='tab:blue')
    ax[5, 0].plot(rm_kd, label='Team Wipes Running Mean', color='tab:blue', alpha=0.25)
    # ax[5, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[5, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    cm_kd = cumulative_mean(np.array(game_df['objectiveTeamWiped']))
    rm_kd = running_mean(np.array(game_df['objectiveTeamWiped']), 50)
    ax[5, 1].set_title('Team Wipes Per Game', fontsize='xx-large')
    ax[5, 1].plot(cm_kd, label='Team Wipes Cumulative Mean', color='tab:blue')
    ax[5, 1].plot(rm_kd, label='Team Wipes Running Mean', color='tab:blue', alpha=0.25)
    # ax[5, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[5, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Revives
    cm_kd = cumulative_mean(np.array(day_df['objectiveReviver']))
    rm_kd = running_mean(np.array(day_df['objectiveReviver']), 50)
    ax[6, 0].set_title('Revives Per Day', fontsize='xx-large')
    ax[6, 0].plot(cm_kd, label='Revives Cumulative Mean', color='tab:blue')
    ax[6, 0].plot(rm_kd, label='Revives Running Mean', color='tab:blue', alpha=0.25)
    # ax[6, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[6, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    cm_kd = cumulative_mean(np.array(game_df['objectiveReviver']))
    rm_kd = running_mean(np.array(game_df['objectiveReviver']), 50)
    ax[6, 1].set_title('Revives Per Game', fontsize='xx-large')
    ax[6, 1].plot(cm_kd, label='Revives Cumulative Mean', color='tab:blue')
    ax[6, 1].plot(rm_kd, label='Revives Running Mean', color='tab:blue', alpha=0.25)
    # ax[6, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[6, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Missions Complete
    cm_kd = cumulative_mean(np.array(day_df['missionsComplete']))
    rm_kd = running_mean(np.array(day_df['missionsComplete']), 50)
    ax[7, 0].set_title('Missions Complete Per Day', fontsize='xx-large')
    ax[7, 0].plot(cm_kd, label='Missions Complete Cumulative Mean', color='tab:blue')
    ax[7, 0].plot(rm_kd, label='Missions Complete Running Mean', color='tab:blue', alpha=0.25)
    # ax[7, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[7, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    cm_kd = cumulative_mean(np.array(game_df['missionsComplete']))
    rm_kd = running_mean(np.array(game_df['missionsComplete']), 50)
    ax[7, 1].set_title('Missions Complete Per Game', fontsize='xx-large')
    ax[7, 1].plot(cm_kd, label='Missions Complete Cumulative Mean', color='tab:blue')
    ax[7, 1].plot(rm_kd, label='Missions Complete Running Mean', color='tab:blue', alpha=0.25)
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
    for i in doc_filter.username_lst:
        temp_df = data[data['uno'] == doc_filter.username_dic[i]]
        people_dic[i] = {j: np.mean(temp_df[j]) for j in col_lst}

    people_df = pd.DataFrame.from_dict(people_dic, orient='index')
    normalized_df = (people_df - people_df.loc['Claim']) / people_df.loc['Claim'] + 1

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, polar=True)

    n = len(doc_filter.username_lst) - 1
    cmap = [plt.get_cmap('viridis')(1. * i / n) for i in range(n)]
    theta = np.linspace(0, 2 * np.pi, len(col_lst) + 1)
    count = 0
    for ind1, person in enumerate(doc_filter.username_lst):
        row = list(normalized_df.loc[person])
        if person == 'Claim':
            ax.plot(theta, row + [row[0]], color='tab:orange', linewidth=4, alpha=1, linestyle=(0, (4, 2)))
        else:
            ax.plot(theta, row + [row[0]], color=cmap[count], linewidth=2, alpha=0.50)
            count += 1
        # Add fill
        #     ax.fill(theta, row + [row[0]], alpha=0.1, color=color)

    ax.legend(labels=doc_filter.username_lst,
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
