import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List
from scipy.stats import norm
from Utils.base import normalize, running_mean, cum_mean


def histo(x: np.ndarray,
          bins: int = 'sturges',
          xlabel: str = 'X Axis',
          xlabel_size: str = 'medium',
          ylabel: str = 'Y Axis',
          ylabel_size: str = 'medium',
          ylabel_color: str = 'tab:orange',
          title: str = 'Histogram',
          title_size: str = 'xx-large',
          label_lst: List[str] = None,
          limit: int = None,
          include_norm: bool = True,
          norm_color: str = 'r',
          norm_ylabel: str = 'Distribution Data',
          norm_lineweight: int = 2,
          norm_legend_location: str = 'lower left',
          color_lst: list = None,
          xtick_rotation: int = -90,
          xtick_size: str = 'small',
          grid: bool = True,
          grid_alpha: float = 0.75,
          grid_lineweight: float = 0.5,
          grid_dash_sequence: tuple = (1, 3),
          legend: bool = True,
          legend_location: str = 'upper right',
          legend_transparency: float = 0.5,
          legend_fontsize: str = 'medium',
          fig_size: tuple = (10, 7),
          stacked: bool = False,
          histtype: str = 'bar',  # {bar, step, stepfilled, barstacked}
          ):
    if limit:
        x = x[:limit]

    if include_norm:
        norm_data = np.random.normal(np.min(x), np.floor(np.sqrt(len(x))), np.max(x))
        _mu, _std = norm.fit(norm_data)

    fig, ax1 = plt.subplots(figsize=fig_size)

    for index in range(x.shape[1]):

        if color_lst:
            color = color_lst[index]
        else:
            color = 'tab:orange'

        if label_lst:
            label = label_lst[index]
        else:
            label = None

        ax1.hist(x[:, index],
                 bins=bins,
                 color=color,
                 label=label,
                 stacked=stacked,
                 histtype=histtype)

    ax1.set_ylabel(ylabel,
                   color=ylabel_color,
                   fontsize=ylabel_size)
    ax1.tick_params(axis='y',
                    labelcolor=ylabel_color)
    ax1.set_title(title,
                  fontsize=title_size)

    if grid:
        ax1.grid(alpha=grid_alpha,
                 linestyle=(0, grid_dash_sequence),
                 linewidth=grid_lineweight)

    ax1.set_xlabel(xlabel,
                   fontsize=xlabel_size)
    ax1.legend(fontsize=legend_fontsize,
               framealpha=legend_transparency,
               loc=legend_location)

    if include_norm:
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin,
                        xmax,
                        100)
        p = norm.pdf(x, _mu, _std)
        l = "Fit Values: mu {:.2f} and std {:.2f}".format(_mu, _std)
        ax2 = ax1.twinx()
        ax2.plot(x,
                 p,
                 color=norm_color,
                 linewidth=norm_lineweight,
                 linestyle='--',
                 label=l)
        ax2.set_ylabel(norm_ylabel,
                       color=norm_color)
        ax2.tick_params(axis='y',
                        labelcolor=norm_color)

        if legend:
            ax2.legend(fontsize=legend_fontsize,
                       framealpha=legend_transparency,
                       loc=norm_legend_location)

    plt.show()


def personal_plot(data: pd.DataFrame, username: str, user_dic: dict, col_lst: List[str], _map: str) -> None:
    uno = user_dic[username]
    base_df = data.iloc[[i for i, j in enumerate(list(data['map'])) if _map in str(j)]]
    base_df = base_df[base_df['uno'] == uno]
    dates = list(base_df['startDate'].unique())

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(30, 30))
    plt.title('Personal Data for: ' + username, fontsize='xx-large')

    # win/loss
    win_count_lst = []
    game_count_lst = []
    wl_ratio_lst = []
    for i in dates:
        temp = base_df[base_df['startDate'] == i]
        wins = len(temp[temp['teamPlacement'] == 1])
        losses = len(temp[temp['teamPlacement'] > 1])
        win_count_lst.append(wins)
        game_count_lst.append(losses + wins)
        wl_ratio_lst.append(wins / (wins + losses))

    wl_df = pd.DataFrame(wl_ratio_lst, columns=['ratio'], index=dates)
    wl_df['wins'] = win_count_lst
    wl_df['losses'] = game_count_lst
    cm_wl = cum_mean(np.array(wl_df['ratio']))
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
    cm_p = cum_mean(np.array(base_df['placementPercent']))
    rm_p = running_mean(np.array(base_df['placementPercent']), 50)
    ax[0, 1].set_title('Team Placement', fontsize='xx-large')
    ax[0, 1].plot(cm_p, label='Placement Cumulative Mean', color='tab:blue')
    ax[0, 1].plot(rm_p, label='Placement Running Mean', color='tab:blue', alpha=0.25)
    ax[0, 1].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[0, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # kd
    cm_kd = cum_mean(np.array(base_df['kdRatio']))
    rm_kd = running_mean(np.array(base_df['kdRatio']), 50)
    ax[1, 0].set_title('Kill Death Ratio', fontsize='xx-large')
    ax[1, 0].plot(cm_kd, label='Kd Ratio Cumulative Mean', color='tab:blue')
    ax[1, 0].plot(rm_kd, label='Kd Ratio Running Mean', color='tab:blue', alpha=0.25)
    ax[1, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[1, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Kills and Deaths
    ax[1, 1].set_title('Kills and Deaths Per Game', fontsize='xx-large')
    cm_kills = cum_mean(np.array(base_df['kills']))
    cm_deaths = cum_mean(np.array(base_df['deaths']))
    rm_kills = running_mean(np.array(base_df['kills']), 50)
    rm_deaths = running_mean(np.array(base_df['deaths']), 50)
    ax[1, 1].set_title('Kills and Deaths', fontsize='xx-large')
    ax[1, 1].plot(cm_kills, label='Kills Cumulative Mean', color='green')
    ax[1, 1].plot(cm_deaths, label='Deaths Cumulative Mean', color='red')
    ax[1, 1].plot(rm_kills, label='Kills Running Mean', color='green', alpha=0.25)
    ax[1, 1].plot(rm_deaths, label='Deaths Running Mean', color='red', alpha=0.25)
    ax[1, 1].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[1, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Damage
    cm_dam_d = cum_mean(np.array(base_df['damageDone']))
    cm_dam_t = cum_mean(np.array(base_df['damageTaken']))
    rm_dam_d = running_mean(np.array(base_df['damageDone']), 50)
    rm_dam_t = running_mean(np.array(base_df['damageTaken']), 50)
    ax[2, 0].set_title('Damage', fontsize='xx-large')
    ax[2, 0].plot(cm_dam_d, label='Damage Done Cumulative Mean', color='green')
    ax[2, 0].plot(cm_dam_t, label='Damage Taken Cumulative Mean', color='red')
    ax[2, 0].plot(rm_dam_d, label='Damage Done Running Mean', color='green', alpha=0.25)
    ax[2, 0].plot(rm_dam_t, label='Damage Taken Running Mean', color='red', alpha=0.25)
    ax[2, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[2, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Misc
    ax[2, 1].set_title('Misc', fontsize='xx-large')
    ax[2, 1].plot(base_df['headshots'])
    ax[2, 1].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[2, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    plt.show()


def lobby_plot(data: pd.DataFrame, _map: str) -> None:
    base_df = data.iloc[[i for i, j in enumerate(list(data['map'])) if _map in str(j)]]
    dates = list(base_df['startDate'].unique())
    games = list(base_df['matchID'].unique())
    col_lst = ['kdRatio', 'kills', 'deaths', 'damageDone', 'damageTaken', 'percentTimeMoving', 'distanceTraveled',
               'objectiveTeamWiped', 'objectiveReviver', 'missionsComplete']

    day_dic = {}
    for date in dates:
        temp_df = base_df[base_df['startDate'] == date].fillna(0)
        day_dic[date] = [np.mean(temp_df[col]) for col in col_lst]

    game_dic = {}
    for game in games:
        temp_df = base_df[base_df['matchID'] == game].fillna(0)
        game_dic[game] = [np.mean(temp_df[col]) for col in col_lst]

    day_df = pd.DataFrame.from_dict(day_dic, orient='index', columns=col_lst).fillna(0)
    game_df = pd.DataFrame.from_dict(game_dic, orient='index', columns=col_lst).fillna(0)

    fig, ax = plt.subplots(nrows=8, ncols=2, figsize=(30, 50))
    plt.title('Lobby Data for', fontsize='xx-large')

    # kd
    cm_kd = cum_mean(np.array(day_df['kdRatio']))
    rm_kd = running_mean(np.array(day_df['kdRatio']), 50)
    ax[0, 0].set_title('Kill Death Ratio Per Day', fontsize='xx-large')
    ax[0, 0].plot(cm_kd, label='Kd Ratio Cumulative Mean', color='tab:blue')
    ax[0, 0].plot(rm_kd, label='Kd Ratio Running Mean', color='tab:blue', alpha=0.25)
    # ax[1, 0].set_xticks(np.arange(min(range(len(day_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[0, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    cm_kd = cum_mean(np.array(game_df['kdRatio']))
    rm_kd = running_mean(np.array(game_df['kdRatio']), 50)
    ax[0, 1].set_title('Kill Death Ratio Per Game', fontsize='xx-large')
    ax[0, 1].plot(cm_kd, label='Kd Ratio Cumulative Mean', color='tab:blue')
    ax[0, 1].plot(rm_kd, label='Kd Ratio Running Mean', color='tab:blue', alpha=0.25)
    # ax[0, 1].set_xticks(np.arange(min(range(len(day_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[0, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Kills and Deaths
    ax[1, 0].set_title('Kills and Deaths Per Game', fontsize='xx-large')
    cm_kills = cum_mean(np.array(day_df['kills']))
    cm_deaths = cum_mean(np.array(day_df['deaths']))
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
    cm_kills = cum_mean(np.array(game_df['kills']))
    cm_deaths = cum_mean(np.array(game_df['deaths']))
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
    cm_dam_d = cum_mean(np.array(day_df['damageDone']))
    cm_dam_t = cum_mean(np.array(day_df['damageTaken']))
    rm_dam_d = running_mean(np.array(day_df['damageDone']), 50)
    rm_dam_t = running_mean(np.array(day_df['damageTaken']), 50)
    ax[2, 0].set_title('Damage Per Day', fontsize='xx-large')
    ax[2, 0].plot(cm_dam_d, label='Damage Done Cumulative Mean', color='green')
    ax[2, 0].plot(cm_dam_t, label='Damage Taken Cumulative Mean', color='red')
    ax[2, 0].plot(rm_dam_d, label='Damage Done Running Mean', color='green', alpha=0.25)
    ax[2, 0].plot(rm_dam_t, label='Damage Taken Running Mean', color='red', alpha=0.25)
    # ax[2, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[2, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    cm_dam_d = cum_mean(np.array(game_df['damageDone']))
    cm_dam_t = cum_mean(np.array(game_df['damageTaken']))
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
    cm_kd = cum_mean(np.array(day_df['percentTimeMoving']))
    rm_kd = running_mean(np.array(day_df['percentTimeMoving']), 50)
    ax[3, 0].set_title('Time Moving Per Day', fontsize='xx-large')
    ax[3, 0].plot(cm_kd, label='Time Moving Cumulative Mean', color='tab:blue')
    ax[3, 0].plot(rm_kd, label='Time Moving Running Mean', color='tab:blue', alpha=0.25)
    # ax[3, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[3, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    cm_kd = cum_mean(np.array(game_df['percentTimeMoving']))
    rm_kd = running_mean(np.array(game_df['percentTimeMoving']), 50)
    ax[3, 1].set_title('Time Moving Per Game', fontsize='xx-large')
    ax[3, 1].plot(cm_kd, label='Time Moving Cumulative Mean', color='tab:blue')
    ax[3, 1].plot(rm_kd, label='Time Moving Running Mean', color='tab:blue', alpha=0.25)
    # ax[3, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[3, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Distance Traveled
    cm_kd = cum_mean(np.array(day_df['distanceTraveled']))
    rm_kd = running_mean(np.array(day_df['distanceTraveled']), 50)
    ax[4, 0].set_title('Distance Traveled Per Day', fontsize='xx-large')
    ax[4, 0].plot(cm_kd, label='Distance Traveled Cumulative Mean', color='tab:blue')
    ax[4, 0].plot(rm_kd, label='Distance Traveled Running Mean', color='tab:blue', alpha=0.25)
    # ax[4, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[4, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    cm_kd = cum_mean(np.array(game_df['distanceTraveled']))
    rm_kd = running_mean(np.array(game_df['distanceTraveled']), 50)
    ax[4, 1].set_title('Distance Traveled Per Game', fontsize='xx-large')
    ax[4, 1].plot(cm_kd, label='Distance Traveled Cumulative Mean', color='tab:blue')
    ax[4, 1].plot(rm_kd, label='Distance Traveled Running Mean', color='tab:blue', alpha=0.25)
    # ax[4, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[4, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Team Wipes
    cm_kd = cum_mean(np.array(day_df['objectiveTeamWiped']))
    rm_kd = running_mean(np.array(day_df['objectiveTeamWiped']), 50)
    ax[5, 0].set_title('Team Wipes Per Day', fontsize='xx-large')
    ax[5, 0].plot(cm_kd, label='Team Wipes Cumulative Mean', color='tab:blue')
    ax[5, 0].plot(rm_kd, label='Team Wipes Running Mean', color='tab:blue', alpha=0.25)
    # ax[5, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[5, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    cm_kd = cum_mean(np.array(game_df['objectiveTeamWiped']))
    rm_kd = running_mean(np.array(game_df['objectiveTeamWiped']), 50)
    ax[5, 1].set_title('Team Wipes Per Game', fontsize='xx-large')
    ax[5, 1].plot(cm_kd, label='Team Wipes Cumulative Mean', color='tab:blue')
    ax[5, 1].plot(rm_kd, label='Team Wipes Running Mean', color='tab:blue', alpha=0.25)
    # ax[5, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[5, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Revives
    cm_kd = cum_mean(np.array(day_df['objectiveReviver']))
    rm_kd = running_mean(np.array(day_df['objectiveReviver']), 50)
    ax[6, 0].set_title('Revives Per Day', fontsize='xx-large')
    ax[6, 0].plot(cm_kd, label='Revives Cumulative Mean', color='tab:blue')
    ax[6, 0].plot(rm_kd, label='Revives Running Mean', color='tab:blue', alpha=0.25)
    # ax[6, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[6, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    cm_kd = cum_mean(np.array(game_df['objectiveReviver']))
    rm_kd = running_mean(np.array(game_df['objectiveReviver']), 50)
    ax[6, 1].set_title('Revives Per Game', fontsize='xx-large')
    ax[6, 1].plot(cm_kd, label='Revives Cumulative Mean', color='tab:blue')
    ax[6, 1].plot(rm_kd, label='Revives Running Mean', color='tab:blue', alpha=0.25)
    # ax[6, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[6, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    # Missions Complete
    cm_kd = cum_mean(np.array(day_df['missionsComplete']))
    rm_kd = running_mean(np.array(day_df['missionsComplete']), 50)
    ax[7, 0].set_title('Missions Complete Per Day', fontsize='xx-large')
    ax[7, 0].plot(cm_kd, label='Missions Complete Cumulative Mean', color='tab:blue')
    ax[7, 0].plot(rm_kd, label='Missions Complete Running Mean', color='tab:blue', alpha=0.25)
    # ax[7, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[7, 0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    cm_kd = cum_mean(np.array(game_df['missionsComplete']))
    rm_kd = running_mean(np.array(game_df['missionsComplete']), 50)
    ax[7, 1].set_title('Missions Complete Per Game', fontsize='xx-large')
    ax[7, 1].plot(cm_kd, label='Missions Complete Cumulative Mean', color='tab:blue')
    ax[7, 1].plot(rm_kd, label='Missions Complete Running Mean', color='tab:blue', alpha=0.25)
    # ax[5, 0].set_xticks(np.arange(min(range(len(base_df['matchID']))), max(range(len(base_df['matchID']))) + 1, 100.0))
    ax[7, 1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)

    plt.show()


def squad_plot(data: pd.DataFrame, username: str, username_lst: List[str], user_dic: dict, col_lst: List[str], _map: str) -> None:
    base_df = data.iloc[[i for i, j in enumerate(list(data['map'])) if _map in str(j)]]

    if username not in username_lst:
        username_lst = [username] + username_lst

    people_dic = {}
    for i in username_lst:
        temp_df = base_df[base_df['uno'] == user_dic[i]]
        people_dic[i] = {j: np.mean(temp_df[j]) for j in col_lst}

    people_df = pd.DataFrame.from_dict(people_dic, orient='index')
    normalized_df = (people_df - people_df.loc['Claim']) / people_df.loc['Claim'] + 1

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, polar=True)

    n = len(username_lst) - 1
    cmap = [plt.get_cmap('viridis')(1. * i / n) for i in range(n)]
    theta = np.linspace(0, 2 * np.pi, len(col_lst) + 1)
    count = 0
    for ind1, person in enumerate(username_lst):
        row = list(normalized_df.loc[person])
        if person == 'Claim':
            ax.plot(theta, row + [row[0]], color='tab:orange', linewidth=4, alpha=1, linestyle=(0, (4, 2)))
        else:
            ax.plot(theta, row + [row[0]], color=cmap[count], linewidth=2, alpha=0.50)
            count += 1
        # Add fill
        #     ax.fill(theta, row + [row[0]], alpha=0.1, color=color)

    ax.legend(labels=username_lst,
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


# def base_plot() -> None:
#     # _map = 'mp_e'
#     # our_data = cod.our_df.iloc[[i for i, j in enumerate(list(cod.our_df['map'])) if _map in str(j)]]
#     # other_data = cod.other_df.iloc[[i for i, j in enumerate(list(cod.other_df['map'])) if _map in str(j)]]
#
#     # day_lst = our_data['startDate'].unique()
#     # day_dic = {i: our_data[our_data['startDate'] == i]['matchID'].unique() for i in day_lst}
#     #
#     # lst = []
#     # for day in day_dic.keys():
#     #     lst.append([np.mean([np.sum(our_data[our_data['matchID'] == i]['kills']) for i in day_dic[day]])] * len(day_dic[day]))
#     # lst = cod.normalize(np.array(sum(lst, [])))
#
#     # match_ids = our_data['matchID'].unique()
#     # new_data = [np.sum(other_data[other_data['matchID'] == i]['kills']) for i in match_ids]
#     # new_our_data = [np.sum(our_data[our_data['matchID'] == i]['kills']) for i in match_ids]
#
#     # Normalize
#     new_data_norm = cod.normalize(np.array(new_data)) * 1
#
#     # Running Mean
#     def running_mean(arr: np.ndarray, num: int) -> np.ndarray:
#         cum_sum = np.cumsum(np.insert(arr=arr, values=[np.mean(arr)] * num, obj=0))
#         return (cum_sum[num:] - cum_sum[:-num]) / float(num)
#
#     new_data_mu = running_mean(arr=np.array(new_data_norm), num=50)
#     hardness_mu = running_mean(arr=np.array(list(hardness_info['difficulty'])), num=50)
#     our_place_mu = running_mean(arr=np.array(list(hardness_info['ourPlacement'])), num=50)
#     our_place_mu2 = running_mean(arr=np.array(list(hardness_info['ourPlacement'])), num=10)
#     our_daily_place_mu = running_mean(arr=np.array(lst), num=50)
#
#     def cum_mean(arr: np.ndarray) -> np.ndarray:
#         cum_sum = np.cumsum(arr, axis=0)
#         for i in range(cum_sum.shape[0]):
#             cum_sum[i] = cum_sum[i] / (i + 1)
#         return cum_sum
#
#     our_place_cum = cum_mean(arr=np.array(list(hardness_info['ourPlacement'])))
## _map = 'mp_e'
# our_data = cod.our_df.iloc[[i for i, j in enumerate(list(cod.our_df['map'])) if _map in str(j)]]
# other_data = cod.other_df.iloc[[i for i, j in enumerate(list(cod.other_df['map'])) if _map in str(j)]]
# day_lst = our_data['startDate'].unique()
#
# match_ids = our_data['matchID'].unique()
# our_lst = [np.mean(our_data[our_data['matchID'] == i]['kdRatio']) for i in match_ids]
# match_hardness_lst = list(hardness_info['difficulty'])
# our_placement_lst = list(hardness_info['ourPlacement'])
#
# #Daily
# our_daily_placement = []
# daily_match_difficulty = []
# daily_compare = []
# for day in day_lst:
#     temp = our_data[our_data['startDate'] == day]
#     temp_n = other_data[other_data['startDate'] == day]
#     length = len(temp['matchID'].unique())
#     our_daily_placement = our_daily_placement + [np.mean(hardness_info.loc[temp['matchID']]['ourPlacement'])] * length
#     daily_match_difficulty = daily_match_difficulty + [np.mean(hardness_info.loc[temp['matchID']]['difficulty'])] * length
#     daily_compare = daily_compare + [np.mean(temp['kdRatio'])] * length
#
# #Running Mean
# rm_match_hardness = running_mean(arr=np.array(match_hardness_lst), num=50)
# rm_our_placement = running_mean(arr=np.array(our_placement_lst), num=50)
# rm_our_lst = running_mean(arr=np.array(our_lst), num=50)
#
# #Cum sum
# cm_match_hardness = cum_mean(arr=np.array(match_hardness_lst))
# cm_our_placement = cum_mean(arr=np.array(our_placement_lst))
# cm_our_lst = cum_mean(arr=np.array(our_lst))


# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 12))
# ax[0].set_xlabel('Matches', fontsize='xx-large')
# # ax[0].set_ylabel('Match Difficulty', fontsize='large', color='tab:blue')
# # # ax[0].plot(normalize(cm_our_lst), label='kd Ratio', color='red', alpha=0.75)
# # # ax[0].plot(normalize(rm_our_lst), label='kd Ratio Rolling Mean (' + str(50) + ')', color='red', alpha=0.25)
# # ax[0].plot(cm_match_hardness, label='Match Difficulty', color='tab:blue', alpha=0.75)
# # ax[0].plot(rm_match_hardness, label='Match Difficulty Rolling Mean (' + str(50) + ')', color='tab:blue', alpha=0.25)
# # ax[0].legend(loc='lower left', fontsize='large', frameon=True, framealpha=0.85)
# # ax2 = ax[0].twinx()
# # ax2.set_ylabel('Our Placement', fontsize='large', color='green')
# ax[0].plot(cm_our_placement, label='Our Placement Cum Mean', color='green', alpha=0.75)
# ax[0].plot(rm_our_placement, label='Our Placement Rolling Mean (' + str(50) + ')', color='green', alpha=0.25)
# ax[0].plot(cm_match_hardness, label='Match Difficulty Cum Mean', color='tab:blue', alpha=0.75)
# ax[0].plot(rm_match_hardness, label='Match Difficulty Rolling Mean (' + str(50) + ')', color='tab:blue', alpha=0.25)
# ax[0].set_xticks(np.arange(min(range(len(match_ids))), max(range(len(match_ids))) + 1, 100.0))
# ax[0].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)
# ax[1].set_xlabel('Matches', fontsize='xx-large')
# # ax[1].set_ylabel('Match Difficulty', fontsize='large', color='tab:blue')
# # ax[1].plot(normalize(daily_compare), label='kd Ratio', color='red', alpha=0.75)
# # ax[1].plot(normalize(rm_our_lst), label='kd Ratio Rolling Mean (' + str(50) + ')', color='red', alpha=0.25)
# # ax[1].plot(daily_match_difficulty, label='Daily Match Difficulty', color='tab:blue', alpha=0.75)
# # ax[1].plot(rm_match_hardness, label='Daily Match Difficulty Rolling Mean (' + str(50) + ')', color='tab:blue', alpha=0.25)
# # ax[1].legend(loc='lower left', fontsize='large', frameon=True, framealpha=0.85)
# # ax2 = ax[0].twiny()
# # ax2.set_ylabel('Our Placement', fontsize='large', color='green')
# ax[1].plot(our_daily_placement, label='Our Daily Placement', color='green', alpha=0.75)
# ax[1].plot(daily_match_difficulty, label='Daily Match Difficulty', color='tab:blue', alpha=0.75)
# # ax2.plot(rm_our_placement, label='Our Daily Placement Rolling Mean (' + str(50) + ')', color='green', alpha=0.25)
# ax[1].set_xticks(np.arange(min(range(len(match_ids))), max(range(len(match_ids))) + 1, 100.0))
# ax[1].legend(loc='lower right', fontsize='large', frameon=True, framealpha=0.85)
# plt.show()

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
# ax.hist(wl_df['wins'], bins='sturges', color='green', stacked=True)
# ax.set_ylabel('Days', fontsize='large')
# ax.set_xlabel('Wins', fontsize='large')
# ax2 = ax.twinx()
#
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin,
#                 xmax,
#                 100)
# norm_data = np.random.normal(np.min(wl_df['wins']),
#                              np.floor(np.sqrt(len(wl_df['wins']))),
#                              np.max(wl_df['wins']))
# mu, _std = norm.fit(norm_data)
# p = norm.pdf(x,
#              mu,
#              _std)
# ax2.plot(x,
#          p)
# plt.show()