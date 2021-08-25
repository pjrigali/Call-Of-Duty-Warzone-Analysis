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
from Classes.regression import regression
from Classes.plot import Line, Scatter, Histogram, Table

from Utils.scrape import connect_to_api, clean_api_data
from Utils.analysis import get_daily_hourly_weekday_stats, first_top5_bottom_stats, bucket_stats
from Utils.analysis import match_difficulty, get_weapons, find_hackers, meta_weapons, previous_next_placement
from Utils.plots import personal_plot, lobby_plot, squad_plot
pd.set_option('display.max_columns', None)


if __name__ == '__main__':
    start_timen = time.time()
    cod = CallofDuty(hacker_data=False, squad_data=True, streamer_mode=True)
    print(''), print('Cod Built'), print("--- %s seconds ---" % round((time.time() - start_timen), 2))

    doc = DocumentFilter(original_df=cod.our_df, map_choice='mp_e', mode_choice='quad')
    doc1 = DocumentFilter(original_df=cod.other_df, map_choice='mp_e', mode_choice='quad')

    # first_top5_bottom_info = first_top5_bottom_stats(doc_filter=doc, col_lst='kills')
    # bucket_info = bucket_stats(doc_filter=doc, placement=[0, 6], col_lst='kills')
    # placement_info = previous_next_placement(doc_filter=doc)
    # match_info = match_difficulty(our_doc_filter=doc, other_doc_filter=doc1)
    # time_info = get_daily_hourly_weekday_stats(doc_filter=doc)
    # weapon_info = get_weapons(doc_filter=doc)
    # hacker_info = find_hackers(doc_filter=doc1, y_column='kdRatio', col_lst=['kills', 'deaths'])
    # meta_weapon_info = meta_weapons(doc_filter=doc, top_5_or_10=True)

    # personal_plot(doc_filter=DocumentFilter(original_df=cod.our_df, map_choice='mp_e', mode_choice='quad',
    #                                         username='Claim', username_dic=cod.name_uno_dict))
    # lobby_plot(doc_filter=doc)
    # squad_plot(doc_filter=DocumentFilter(original_df=cod.our_df, map_choice='mp_e', mode_choice='quad',
    #                                      username_lst=cod.user.squad, username_dic=cod.name_uno_dict),
    #            col_lst=['kills', 'deaths', 'kdRatio', 'headshots'])

    # table_data = [
    #     ["matched gt", 10],
    #     ["unmatched gt", 20],
    #     ["total gt", 30],
    #     ["mean_precision", 0.6],
    #     ["mean_recall", 0.4]
    # ]
    # t = Table(data=pd.DataFrame(table_data), header_colors='tab:blue', sequential_cells=True)
    # plt.show()
    cod
