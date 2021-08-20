from typing import Optional, List
from dataclasses import dataclass
from credentials import user_inputs


@dataclass
class User:

    _file_name: str
    _repo: str
    _gamertag: str
    _squad: List[str]

    # Need for Scraping
    headers: Optional[dict]
    CodTrackerID: Optional[str]
    USERNAME: Optional[str]
    PASSWORD: Optional[str]
    DRIVER_PATH: Optional[str]

    def __init__(self, info: dict = None):

        if info is None:
            info = user_inputs

        if info['file_name'] is None:
            raise AttributeError('Need to pass a file name')
        else:
            self._file_name: str = info['file_name']

        if info['repo'] is None:
            raise AttributeError('Need to pass a repo directory')
        else:
            self._repo: str = info['repo']

        if info['gamertag'] is None:
            raise AttributeError('Need to pass a gamertag')
        else:
            self._gamertag: str = info['gamertag']

        if info['squad'] is None:
            raise AttributeError('Need to pass a list of gamertags')
        else:
            self._squad: List[str] = info['squad']

        # Need for Scraping
        if info['headers'] is None:
            self.headers = {'Needed for scraping': 'Needed for scraping'}
        else:
            self.headers = info['headers']

        if info['codtrackerid'] is None:
            self.CodTrackerID = 'Needed for scraping'
        else:
            self.CodTrackerID = info['codtrackerid']

        if info['username'] is None:
            self.USERNAME = 'Needed for scraping'
        else:
            self.USERNAME = info['username']

        if info['password'] is None:
            self.PASSWORD = 'Needed for scraping'
        else:
            self.PASSWORD = info['password']

        if info['driverpath'] is None:
            self.DRIVER_PATH = 'Needed for scraping'
        else:
            self.DRIVER_PATH = info['driverpath']

    def __repr__(self):
        return self.gamertag

    @property
    def file_name(self) -> str:
        return self._file_name

    @property
    def repo(self) -> str:
        return self._repo

    @property
    def gamertag(self) -> str:
        return self._gamertag

    @property
    def squad(self) -> List[str]:
        return self._squad
