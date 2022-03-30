from dataclasses import dataclass
from warzone.classes.document_filter import DocumentFilter
from warzone.utils.class_functions import _build_windows


@dataclass
class TSWindows:

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
        return 'TSWindow'