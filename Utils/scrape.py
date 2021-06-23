# from selenium.webdriver.common.by import By
from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
import datetime
import time
import requests
import json
import pandas as pd
pd.set_option('display.max_columns', None)


def get_cod_data_api(last_match_timestamp,
                     seconds: int = 5,
                     CodTrackerID: str = 'Prigali#1499',
                     DRIVER_PATH: str = '/Users/Peter/Desktop/chromedriver',
                     ) -> list:
    
    driver = webdriver.Chrome(executable_path=DRIVER_PATH)
    driver.implicitly_wait(seconds)
    driver.get('https://cod.tracker.gg/')

    select_battle_net = driver.find_element_by_xpath('//*[@id="app"]/div[2]/div[2]/div/main/div[2]/div[1]/div[1]/div/div/div[1]/div/div/div[1]/ul/li[3]')
    select_battle_net.click()

    search_users = driver.find_element_by_xpath('//*[@id="app"]/div[2]/div[2]/div/main/div[2]/div[1]/div[1]/div/div/div[1]/div/div/div[1]/form/input')
    search_users.send_keys([CodTrackerID])
    time.sleep(seconds)

    select_prigali = driver.find_element_by_xpath('//*[@id="app"]/div[2]/div[2]/div/main/div[2]/div[1]/div[1]/div/div/div[1]/div/div/div[1]/form/div/div/div')
    select_prigali.click()
    time.sleep(seconds)

    select_matches = driver.find_element_by_xpath('/html/body/div/div[2]/div[2]/div/main/div[2]/div[1]/div[3]/ul/li[3]')
    select_matches.click()

    # month_dict1 = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    month_dict2 = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    
    if type(last_match_timestamp) == str:
        element = datetime.datetime.strptime(last_match_timestamp, "%Y-%m-%d")# '2021-06-11'
    else:
        element = last_match_timestamp
        
    element_int = int(str(element.month) + str(element.day))

    while True:
        time.sleep(5)
        load_page = driver.find_element_by_xpath('.//button[contains(@class,"trn-button")]')
        load_page.click()
        dates = driver.find_elements_by_xpath('.//div[contains(@class,"session-header__title")]')
        date_lst = [date.text.split("\n")[0] for date in dates]
        latest_month, latest_date = date_lst[-1].split(' ')[0], date_lst[-1].split(' ')[1]
        latest = int(str(month_dict2[latest_month]) + latest_date)
        
        if element_int > latest:
            print(element_int, latest, '...complete')
            break
        else:
            print(element_int, latest)

    ml = driver.find_elements_by_xpath('//a[contains(@class,"match-row__link")]')
    lobby_links = [i.get_attribute('href') for i in ml]
    driver.close()
    
    return [i.split('/')[5].split('?')[0] for i in lobby_links]


def connect_to_api(_id: str):
    url = "https://www.callofduty.com/api/papi-client/crm/cod/v2/title/mw/platform/battle/fullMatch/wz/{}/it".format(
        _id)
    response = requests.request("GET", url, headers={})
    return json.loads(response.text)['data']['allPlayers']


def clean_api_data(_json) -> pd.DataFrame:
    base = []
    for team in _json:
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


def refresh_data(old_df: pd.DataFrame,
                 last_match_timestamp,
                 repo: str,
                 filename: str = None,
                 savedf: bool = True
                 ) -> pd.DataFrame:
    
    old_id_set = {str(i): True for i in set(old_df['matchID'])}
    new_id_lst = get_cod_data_api(last_match_timestamp)
    lst_to_collect = [i for i in new_id_lst if i not in old_id_set]
    
    lst_of_df, count = [], len(lst_to_collect)
    for _id in lst_to_collect:
        lst_of_df.append(clean_api_data(connect_to_api(_id=_id)))
        count -= 1
        time.sleep(1)
        print(count, '...left to go')
    
    comb_df = pd.concat([old_df, pd.concat(lst_of_df).reset_index(drop=True)]).reset_index(drop=True)
    
    if filename is None:
        filename = str(datetime.datetime.today()) + '_Personal_Match_Data'

    if savedf:
        comb_df.to_csv(repo+'{}.csv'.format(filename))

    return comb_df



