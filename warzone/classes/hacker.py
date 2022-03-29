from dataclasses import dataclass
from warzone.classes.user import User
from warzone.utils.class_functions import evaluate_df, uno_username_dict, get_hacker_and_other_df


@dataclass
class Hacker:

    __slots__ = ['whole', 'name_uno_dic', 'our_df', 'other_df']

    def __init__(self,
                 user: User,
                 build_json: bool = False,
                 from_json: bool = False,
                 reset_dtype: bool = False):

        self.whole = evaluate_df(file_name=user.hacker_file_name, repo=user.hacker_repo, json_path=user.hacker_json_repo,
                                 build_json=build_json, from_json=from_json, reset_dtype=reset_dtype)
        self.name_uno_dic = uno_username_dict(data=self.whole)
        self.our_df, self.other_df = get_hacker_and_other_df(data=self.whole, min_count=10)

    def __repr__(self):
        return 'HackerData'
