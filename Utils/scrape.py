import requests
import json
import pandas as pd


def connect_to_api(_id: str):
    url = "https://www.callofduty.com/api/papi-client/crm/cod/v2/title/mw/platform/battle/fullMatch/wz/{}/it".format(
        _id)
    response = requests.request("GET", url, headers={})
    return json.loads(response.text)['data']['allPlayers']


def clean_api_data(json_object) -> pd.DataFrame:
    base = []
    for team in json_object:
        player_data = {key: team[key] for key in team.keys() if key not in ['playerStats', 'player']}

        for key in team['playerStats'].keys():
            player_data[key] = team['playerStats'][key]

        for key in team['player'].keys():

            if key == 'brMissionStats':
                for i in team['player']['brMissionStats'].keys():
                    if i == 'missionStatsByType':
                        for j in team['player']['brMissionStats'][i].keys():
                            for k in team['player']['brMissionStats'][i][j].keys():
                                player_data[j + '_' + k] = team['player']['brMissionStats'][i][j][k]
                    else:
                        player_data[i] = team['player']['brMissionStats'][i]
            elif key == 'loadout':
                count = 1
                for i in team['player']['loadout']:
                    player_data['primaryWeapon_' + str(count)] = i['primaryWeapon']['name']
                    player_data['primaryWeaponAttachements_' + str(count)] = [k['name'] for k in
                                                                              i['primaryWeapon']['attachments']]
                    player_data['secondaryWeapon_' + str(count)] = i['secondaryWeapon']['name']
                    player_data['secondaryWeaponAttachements_' + str(count)] = [k['name'] for k in
                                                                                i['secondaryWeapon']['attachments']]
                    player_data['perks_' + str(count)] = [k['name'] for k in i['perks']]
                    player_data['extraPerks_' + str(count)] = [k['name'] for k in i['extraPerks']]
                    player_data['killstreaks_' + str(count)] = [k['name'] for k in i['killstreaks']]
                    player_data['tactical_' + str(count)] = i['tactical']['name']
                    player_data['lethal_' + str(count)] = i['lethal']['name']
                    count += 1
            else:
                player_data[key] = team['player'][key]
        base.append(player_data)
    return pd.DataFrame(base)
