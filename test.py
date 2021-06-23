import pandas as pd
import datetime
from ast import literal_eval
pd.set_option('display.max_columns', None)

link = 'C:\\Users\\Peter\\Desktop\\Personal\\11_Repository\\Call of Duty Related\\Data\\Personal Data\\Personal_Match_Data.csv'
df = pd.read_csv(link, index_col='Unnamed: 0')
remove_columns = ['playlistName', 'version', 'gameType', 'rankedTeams', 'draw', 'privateMatch', 'rank', 'nearmisses',
                  'awards', 'killsteaks']
day_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}


def convert_to_str(text):
    return str(text)


def convert_to_date_or_time(text, index: int = None, str_output: bool = True):
    if str_output:
        return str(datetime.datetime.fromtimestamp(text).strftime('%Y-%m-%d %H:%M').split(" ", 1)[index])
    else:
        return datetime.datetime.strptime(text, "%Y-%m-%d")
    
    
df['dateTime'] = (df['date'] + ' ' + df['time']).astype(str)
df['startDate'] = [convert_to_date_or_time(i, 0) for i in df['utcStartSeconds']]
df['startTime'] = [convert_to_date_or_time(i, 1) for i in df['utcStartSeconds']]
df['endDate'] = [convert_to_date_or_time(i, 0) for i in df['utcEndSeconds']]
df['endTime'] = [convert_to_date_or_time(i, 1) for i in df['utcEndSeconds']]
df['weekday'] = [day_dic[convert_to_date_or_time(text=i, str_output=False).date().weekday()] for i in df['date']]
df['percentTimeMoving'] = [str(round(i, 2)) for i in df['percentTimeMoving']]
df['placementPercent'] = (1 - df['teamPlacement'] / df['teamCount']).round(2)

# Convert to str
column_lst = ['map', 'mode', 'matchID', 'duration', 'playerCount', 'teamCount', 'kills', 'medalXp',
              'objectiveTeamWiped', 'objectiveLastStandKill', 'matchXp', 'scoreXp', 'wallBangs', 'score', 'totalXp',
              'headshots', 'assists', 'challengeXp', 'scorePerMinute', 'distanceTraveled', 'teamSurvivalTime',
              'deaths', 'objectiveMunitionsBoxTeammateUsed', 'objectiveBrDownEnemyCircle3', 'kdRatio',
              'objectiveBrDownEnemyCircle2', 'objectiveBrMissionPickupTablet', 'bonusXp', 'objectiveBrKioskBuy',
              'gulagDeaths', 'timePlayed', 'executions', 'gulagKills', 'objectiveBrCacheOpen', 'miscXp',
              'longestStreak', 'teamPlacement', 'damageDone', 'damageTaken', 'team', 'username', 'uno',
              'missionsComplete', 'totalMissionXpEarned', 'totalMissionWeaponXpEarned',
              ]

for col in column_lst:
    df[col] = [convert_to_str(i) for i in df[col]]

for gun in ['primaryWeapon', 'secondaryWeapon']:
    gun_name_lst = []
    attachments = []
    for i in df[gun]:
        temp_dict = literal_eval(i)
        if temp_dict != 0:
            gun_name_lst.append(temp_dict['name'])
            attachments.append(' '.join([i['name'] for i in temp_dict['attachments'] if i['name'] is not None]))
        else:
            gun_name_lst.append(' ')
            attachments.append(' ')
    df[gun+'Name'] = gun_name_lst
    df[gun+'Attachments'] = attachments
    
for perk in ['perks', 'extraPerks']:
    perk_lst = []
    for i in df[perk]:
        temp_dict = literal_eval(i)
        if temp_dict != 0:
            perk_lst.append(' '.join([i['name'] for i in temp_dict if i['name'] is not None]))
        else:
            perk_lst.append(' ')
    df[perk] = perk_lst
    
for equip in ['tactical', 'lethal']:
    equip_lst = []
    for i in df[equip]:
        temp_dict = literal_eval(i)
        if temp_dict != 0:
            equip_lst.append(temp_dict['name'])
        else:
            equip_lst.append(' ')
    df[equip] = equip_lst
    
mission_lst = []
for mission in df['missionStatsByType']:
    temp_dict = literal_eval(mission)
    missions = ' '
    if temp_dict != 0:
        missions = ' '.join([i+str(int(temp_dict[i]['count'])) for i in temp_dict.keys()])
    mission_lst.append(missions)
df['missions'] = mission_lst

new_cols = ['dateTime', 'startDate', 'startTime', 'endDate', 'endTime', 'weekday', 'percentTimeMoving',
            'placementPercent', 'primaryWeaponName', 'primaryWeaponAttachments', 'secondaryWeaponName',
            'secondaryWeaponAttachments', 'perks', 'extraPerks', 'tactical', 'lethal', 'missions']

dfn = df[column_lst + new_cols]
link = 'C:\\Users\\Peter\\Desktop\\Personal\\11_Repository\\Call of Duty Related\\Data\\Personal Data\\Personal_Match_Data_Prebuilt.csv'
dfn.to_csv(link)
# dfn
# df_open = pd.read_csv('C:\\Users\\Peter\\Desktop\\Personal\\11_Repository\\Call of Duty Related\\Data\\Personal Data\\Personal_Match_Data_Prebuilt.csv',
#                       index_col='Unnamed: 0')
#                       # dtype={'map': str,
#                       #                                'mode': str,
#                       #                                'matchID': str,
#                       #                                'duration': int,
#                       #                                'playerCount': int,
#                       #                                'teamCount': int,
#                       #                                'kills': int,
#                       #                                'medalXp': int,
#                       #                                'objectiveTeamWiped': int,
#                       #                                'objectiveLastStandKill': int,
#                       #                                'matchXp': int,
#                       #                                'scoreXp': int,
#                       #                                'wallBangs': int,
#                       #                                'score': int,
#                       #                                'totalXp': int,
#                       #                                'headshots': int,
#                       #                                'assists': int,
#                       #                                'challengeXp': int,
#                       #                                'scorePerMinute': float,
#                       #                                'distanceTraveled': float,
#                       #                                'teamSurvivalTime': int,
#                       #                                'deaths': int,
#                       #                                'objectiveMunitionsBoxTeammateUsed': int,
#                       #                                'objectiveBrDownEnemyCircle3': int,
#                       #                                'kdRatio': float,
#                       #                                'objectiveBrDownEnemyCircle2': int,
#                       #                                'objectiveBrMissionPickupTablet': int,
#                       #                                'bonusXp': int,
#                       #                                'objectiveBrKioskBuy': int,
#                       #                                'gulagDeaths': int,
#                       #                                'timePlayed': int,
#                       #                                'executions': int,
#                       #                                'gulagKills': int,
#                       #                                'objectiveBrCacheOpen': int,
#                       #                                'miscXp': int,
#                       #                                'longestStreak': int,
#                       #                                'teamPlacement': int,
#                       #                                'damageDone': int,
#                       #                                'damageTaken': int,
#                       #                                'team': str,
#                       #                                'username': str,
#                       #                                'uno': str,
#                       #                                'missionsComplete': int,
#                       #                                'totalMissionXpEarned': int,
#                       #                                'totalMissionWeaponXpEarned': int,
#                       #                                'startDate': str,
#                       #                                'startTime': str,
#                       #                                'endDate': str,
#                       #                                'endTime': str,
#                       #                                'weekday': str,
#                       #                                'percentTimeMoving': float,
#                       #                                'primaryWeaponName': str,
#                       #                                'primaryWeaponAttachments': str,
#                       #                                'secondaryWeaponName': str,
#                       #                                'secondaryWeaponAttachments': str,
#                       #                                'perks': str,
#                       #                                'extraPerks': str,
#                       #                                'tactical': str,
#                       #                                'lethal': str,
#                       #                                'missions': str
#                       #                                }
# df_open
