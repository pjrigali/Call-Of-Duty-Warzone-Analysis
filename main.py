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
pd.set_option('display.max_columns', None)
from Classes.document_filter import DocumentFilter
from Classes.call_of_duty import CallofDuty

from Utils.gun_dictionary import gun_dict
from Utils.scrape_temp import refresh_data, connect_to_api, clean_api_data
from Utils.base import running_mean, cumulative_mean, normalize
from Utils.regress import regression_calcs, regress
from Utils.analysis import get_daily_hourly_weekday_stats
from Utils.analysis import placement_descriptive_stats, first_top5_bottom_stats, bucket, previous_next_placement
from Utils.analysis import weekly_stats, daily_stats, match_difficulty, get_weapons, find_hackers, meta_weapons
from Utils.plots import personal_plot, lobby_plot, squad_plot
from Utils.medium_posts import deaths_per_circle, engagement_mm, hackers_overtime, find_hackers_from_hacker_df, squad_effect


if __name__ == '__main__':
    start_timen = time.time()
    cod = CallofDuty(hacker_data=False, squad_data=False)
    print(''), print('Cod Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))
    cod

