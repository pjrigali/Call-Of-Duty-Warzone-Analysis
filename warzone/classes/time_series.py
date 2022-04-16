"""User class object.

Usage:
 ./warzone/classes/time_series.py

Author:
 Peter Rigali - 2021-08-30
"""
from dataclasses import dataclass
import pandas as pd
from warzone.classes.document_filter import DocumentFilter
from warzone.classes.tswindow import TSWindows
from warzone.analysis import match_difficulty
from warzone.utils.class_functions import _build_sessions
from warzone.utils.lists import SUM_LST
from pyjr.classes.data import Data


@dataclass
class TSanalysis:
    """

    Builds TSanalysis Class. Used for time series analysis of matches.

    :param our_doc_filter: Input DocumentFilter of players data.
    :type our_doc_filter: DocumentFilter
    :param other_doc_filter: Input DocumentFilter of other players data.
    :type other_doc_filter: DocumentFilter
    :param difficulty: If True, will calculate the match difficulty.
    :type difficulty: bool
    :example: *None*
    :note: *None*

    """
    __slots__ = ('our_doc', 'other_doc', 'windows', 'match_difficulty', 'our_sessions', 'other_sessions',
                 'average_session', 'average_window')

    def __init__(self, our_doc_filter: DocumentFilter, other_doc_filter: DocumentFilter, difficulty: bool = False):
        self.our_doc = our_doc_filter
        self.other_doc = other_doc_filter
        self.windows = None
        self.average_window = None

        self.match_difficulty = None
        if difficulty:
            self.match_difficulty = match_difficulty(our_doc_filter=our_doc_filter, other_doc_filter=other_doc_filter)

        self.our_sessions = None
        self.other_sessions = None
        self.average_session = None

    def add_sessions(self, session_type: str = 'session', session_value: int = 60):
        """Builds sessions outside of windows"""
        self.our_sessions, self.other_sessions = _build_sessions(our_df=self.our_doc.df, other_df=self.other_doc.df,
                                                                 session_type=session_type, session_value=session_value)
        return self

    def add_windows(self, stat_type: str = 'mean', session_type: str = 'session', session_value: int = 60):
        """Builds windows from DocumentFilters."""
        self.windows = TSWindows(our_doc_filter=self.our_doc, other_doc_filter=self.other_doc, stat_type=stat_type,
                                 session_type=session_type, session_value=session_value)
        return self

    def get_window_df(self) -> pd.DataFrame:
        """Return windows as DataFrame."""
        lst = []
        for window in self.windows.windows:
            for game in window.games:
                tm, lo = game.get_dict()
                tm['matchID'], tm['window'], tm['game'], tm['from'] = game.match_id, window.position, game.position, 'team'
                lo['matchID'], lo['window'], lo['game'], lo['from'] = game.match_id, window.position, game.position, 'lobby'
                lst.append(tm), lst.append(lo)
        return pd.DataFrame(lst).sort_values(['window', 'game'], ascending=True)

    def get_session_df(self) -> pd.DataFrame:
        """Return sessions as DataFrame."""
        lst = []
        for key, val in self.our_sessions.items():
            val['session'] = key
            lst.append(val)
        return pd.concat(lst, axis=1)

    def add_ave_window(self):
        """Calc stats for all games within each window."""
        self.windows.add_average()
        self.average_window = self.windows.average_window
        return self

    def add_ave_session(self):
        """Calc stats for all games within each session"""
        # Needs to keep match in series and calc stats based on series position
        dic = {i: [] for i in SUM_LST}
        for key, val in self.our_sessions.items():
            for col in SUM_LST:
                for i in val[col]:
                    dic[col].append(i)
                # for i in self.other_sessions[key][col]:
                #     dic[col].append(i)
        for key, val in dic.items():
            dic[key] = Data(data=val, name=key, na_handling='zero')
        self.average_session = dic
        return self

    def __repr__(self):
        return 'TSanalysis'
