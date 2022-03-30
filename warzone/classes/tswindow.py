from dataclasses import dataclass
from warzone.classes.document_filter import DocumentFilter
from warzone.utils.class_functions import _build_windows


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
    __slots__ = ('windows', 'stat_type', 'session_type', 'session_value', 'len')

    def __init__(self, our_doc_filter: DocumentFilter, other_doc_filter: DocumentFilter,
                 session_type: str = 'session', session_value: int = 60, stat_type: str = 'sum'):
        self.windows = _build_windows(our_df=our_doc_filter.df, other_df=other_doc_filter.df, session_type=session_type,
                                      session_value=session_value, stat_type=stat_type)
        self.stat_type = stat_type
        self.session_type = session_type
        self.session_value = session_value
        self.len = self.windows.__len__()

    def __repr__(self):
        return 'TSWindows'
