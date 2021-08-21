import pandas as pd
from credentials import user_inputs
from Classes.user import User
from Classes.squad import Squad
from Utils.build import evaluate_df, get_our_and_other_df, get_match_id_set
from Utils.gun_dictionary import gun_dict


class CallofDuty:

    def __init__(self, hacker_data: bool = False, squad_data: bool = False):
        self._User = User(info=user_inputs)
        self._whole: pd.DataFrame = evaluate_df(file_name=self._User.file_name, repo=self._User.repo)
        self._gun_dic = gun_dict
        self._last_match_date_time = list(self._whole['startDateTime'])[-1]
        self._name_uno_dict = get_match_id_set(data=self._whole)
        self._my_uno = self.name_uno_dict[self._User.gamertag]
        self._our_df, self._other_df = get_our_and_other_df(data=self._whole, _my_uno=self._my_uno,
                                                            squad_name_lst=self._User.squad,
                                                            name_uno_dict=self._name_uno_dict)
        self._hacker_df = None
        self._name_uno_dict_hacker = None
        if hacker_data:
            self._hacker_df = evaluate_df(file_name='hacker_df.csv', repo=self._User.repo)
            self._name_uno_dict_hacker = get_match_id_set(data=self.hacker_df)

        self._Squad = None
        if squad_data:
            self._Squad = Squad(squad_lst=self._User.squad, original_df=self.our_df, uno_name_dic=self.name_uno_dict)

    def __repr__(self):
        return 'Call of Duty'

    @property
    def whole(self) -> pd.DataFrame:
        return self._whole

    @property
    def gun_dictionary(self) -> pd.DataFrame:
        return self._gun_dic

    @property
    def last_match_date_time(self) -> pd.DataFrame:
        return self._last_match_date_time

    @property
    def name_uno_dict(self) -> pd.DataFrame:
        return self._name_uno_dict

    @property
    def my_uno(self) -> pd.DataFrame:
        return self._my_uno

    @property
    def our_df(self) -> pd.DataFrame:
        return self._our_df

    @property
    def other_df(self) -> pd.DataFrame:
        return self._other_df

    @property
    def hacker_df(self) -> pd.DataFrame:
        return self._hacker_df

    @property
    def name_uno_dict_hacker(self) -> pd.DataFrame:
        return self._name_uno_dict_hacker

    @property
    def user(self) -> pd.DataFrame:
        return self._User

    @property
    def squad(self) -> pd.DataFrame:
        return self._Squad
