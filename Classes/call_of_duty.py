import pandas as pd
from credentials import user_inputs
from Classes.user import User
from Classes.squad import Squad
from Utils.build import _sm_whole, _sm_gamertags, _evaluate_df, _get_our_and_other_df, _get_match_id_set
from Utils.gun_dictionary import gun_dict


class CallofDuty:
    """
    Calculate stats for all maps/modes for each squad memeber.

    Parameters
    ----------
    hacker_data : bool, default is False.
        This Requires a seperate csv with hacker data saved. This data can be collected by finding hackers after
        the fact and scraping there data from CodTracker, this can then be used to find hackers in other games.
    squad_data : bool, default is False.
        If True, will build the Squad class.

    Examples
    --------

    >>> from Classes.call_of_duty import CallofDuty
    >>> cod = CallofDuty(hacker_data=False, squad_data=True)

    This will calculate and build the CallofDuty class.
    """

    def __init__(self, hacker_data: bool = False, squad_data: bool = False, streamer_mode: bool = True):
        self._User = User(info=user_inputs)
        self._whole: pd.DataFrame = _evaluate_df(file_name=self._User.file_name, repo=self._User.repo)

        if streamer_mode:
            _sm_whole(_user_class=self._User, data=self._whole)

        self._gun_dic = gun_dict
        self._last_match_date_time = list(self._whole['startDateTime'])[-1]
        self._name_uno_dict = _get_match_id_set(data=self._whole)

        if streamer_mode:
            _sm_gamertags(_user=self._User)

        self._my_uno = self.name_uno_dict[self._User.gamertag]
        self._our_df, self._other_df = _get_our_and_other_df(data=self._whole, _my_uno=self._my_uno,
                                                             squad_name_lst=self._User.squad,
                                                             name_uno_dict=self._name_uno_dict)
        self._hacker_df = None
        self._name_uno_dict_hacker = None
        if hacker_data:
            self._hacker_df = _evaluate_df(file_name='hacker_df.csv', repo=self._User.repo)
            self._name_uno_dict_hacker = _get_match_id_set(data=self.hacker_df)

        self._Squad = None
        if squad_data:
            self._Squad = Squad(squad_lst=self._User.squad, original_df=self.our_df, uno_name_dic=self.name_uno_dict)

    def __repr__(self):
        return 'Call of Duty'

    @property
    def whole(self) -> pd.DataFrame:
        """Returns the unedited player matches DataFrame"""
        return self._whole

    @property
    def gun_dictionary(self) -> dict:
        """Returns a dict of gun names"""
        return self._gun_dic

    @property
    def last_match_date_time(self):
        """Returns a Timestamp of the latest game in the players data. Useful when scraping from Cod Tracker"""
        return self._last_match_date_time

    @property
    def name_uno_dict(self) -> dict:
        """Returns a dict of gamertags and respective unos"""
        return self._name_uno_dict

    @property
    def my_uno(self) -> str:
        """Returns the user uno value"""
        return self._my_uno

    @property
    def our_df(self) -> pd.DataFrame:
        """Returns a DataFrame of all data related to player and there teammates"""
        return self._our_df

    @property
    def other_df(self) -> pd.DataFrame:
        """Returns a DataFrame of all data related to other teams in a lobby"""
        return self._other_df

    @property
    def hacker_df(self) -> pd.DataFrame:
        """If a hacker DataFrame is provided, will return just the hacker DataFrame"""
        return self._hacker_df

    @property
    def name_uno_dict_hacker(self) -> dict:
        """If a hacker DataFrame is provided, will return the gamertags: unos for the hacker DataFrame"""
        return self._name_uno_dict_hacker

    @property
    def user(self) -> User:
        """Returns a User class object of related info to the user"""
        return self._User

    @property
    def squad(self) -> Squad:
        """Returns a Squad class object of stats related to the user squad mates"""
        return self._Squad
