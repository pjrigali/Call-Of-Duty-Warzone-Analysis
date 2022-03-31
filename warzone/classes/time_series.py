"""User class object.

Usage:
 ./warzone/classes/time_series.py

Author:
 Peter Rigali - 2021-08-30
"""
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
from warzone.classes.document_filter import DocumentFilter
from warzone.classes.tswindow import TSWindows
from warzone.analysis import match_difficulty
from warzone.utils.class_functions import _build_sessions


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
    __slots__ = ('our_doc', 'other_doc', 'windows', 'match_difficulty', 'our_sessions', 'other_sessions')

    def __init__(self, our_doc_filter: DocumentFilter, other_doc_filter: DocumentFilter, difficulty: bool = False):
        self.our_doc = our_doc_filter
        self.other_doc = other_doc_filter
        self.windows = None

        self.match_difficulty = None
        if difficulty:
            self.match_difficulty = match_difficulty(our_doc_filter=our_doc_filter, other_doc_filter=other_doc_filter)

        self.our_sessions = None
        self.other_sessions = None

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

    def __repr__(self):
        return 'TSanalysis'
