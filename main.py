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
from Classes.document_filter import DocumentFilter
from Classes.call_of_duty import CallofDuty
from Utils.scrape_temp import connect_to_api, clean_api_data
from Utils.base import running_mean, cumulative_mean, normalize
from Utils.regress import regression_calcs, regress
from Utils.analysis import get_daily_hourly_weekday_stats
from Utils.analysis import placement_descriptive_stats, first_top5_bottom_stats, bucket, previous_next_placement
from Utils.analysis import weekly_stats, daily_stats, match_difficulty, get_weapons, find_hackers, meta_weapons
from Utils.plots import personal_plot, lobby_plot, squad_plot
from Utils.medium_posts import deaths_per_circle, engagement_mm, hackers_overtime, find_hackers_from_hacker_df, squad_effect
from Classes.plot import Line, Scatter, Histogram
pd.set_option('display.max_columns', None)


if __name__ == '__main__':
    start_timen = time.time()
    cod = CallofDuty(hacker_data=False, squad_data=False)
    print(''), print('Cod Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))

    doc = DocumentFilter(original_df=cod.our_df, map_choice='mp_e', mode_choice='quad')
    # doc1 = DocumentFilter(original_df=cod.other_df, map_choice='mp_e', mode_choice='quad')

    # time_info = get_daily_hourly_weekday_stats(doc_filter=doc)
    # weapon_info = get_weapons(doc_filter=doc)
    # meta_weapon_info = meta_weapons(doc_filter=doc, top_5_or_10=True)
    # daily_info = daily_stats(doc_filter=doc)
    # weekly_info = weekly_stats(doc_filter=doc)
    # hacker_info = find_hackers(doc_filter=doc1, y_column='kdRatio', col_lst=[])
    # match_info = match_difficulty(our_doc_filter=doc, other_doc_filter=doc1)
    cod

