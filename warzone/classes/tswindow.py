"""TSWindows class object.

Usage:
 ./warzone/classes/tswindow.py

Author:
 Peter Rigali - 2022-03-30
"""
from dataclasses import dataclass
from warzone.classes.document_filter import DocumentFilter
from warzone.utils.class_functions import _build_windows
from warzone.utils.lists import SUM_LST
from pyjr.classes.data import Data
from pyjr.utils.tools.math import _perc


@dataclass
class TSWindows:
    """
    TSWindows class builds windows based on desired input session.

    :param our_doc_filter: Input DocumentFilter for our.
    :type our_doc_filter: DocumentFilter
    :param other_doc_filter: Input DocumentFilter for other.
    :type other_doc_filter: DocumentFilter
    :param session_type: Must be {session, game, event, or day}
    :type session_type: str
    :param session_value: Window width, only needed when session_type is session, event or game.
    :type session_value: int
    :param stat_type: Desired stat to be calculated.
    :type stat_type: str
    """
    __slots__ = ['windows', 'stat_type', 'session_type', 'session_value', 'len', 'average_window']

    def __init__(self, our_doc_filter: DocumentFilter, other_doc_filter: DocumentFilter,
                 session_type: str = 'session', session_value: int = 60, stat_type: str = 'sum'):
        self.windows = _build_windows(our_df=our_doc_filter.df, other_df=other_doc_filter.df, session_type=session_type,
                                      session_value=session_value, stat_type=stat_type)
        self.stat_type = stat_type
        self.session_type = session_type
        self.session_value = session_value
        self.len = self.windows.__len__()
        self.average_window = None

    def add_average(self):
        m = int(_perc(d=[window.len for window in self.windows], q=0.95))
        check = {i: True for i in range(0, m)}
        dic = {i: {i: [] for i in SUM_LST} for i in range(m)}
        for window in self.windows:
            for game in window.games:
                tm, lo = game.get_dict()
                for key, val in tm.items():
                    if game.position in check:
                        dic[game.position][key].append(val)

        for key, val in dic.items():
            for key1, val1 in val.items():
                dic[key][key1] = Data(data=val1, name=key1, na_handling='zero')
        self.average_window = dic
        return self

    def __repr__(self):
        return 'TSWindows'
