from typing import Optional, List
from dataclasses import dataclass
from credentials import user_inputs


@dataclass
class User:

    file_name: str
    USERNAME: str
    PASSWORD: str
    DRIVER_PATH: Optional[str]
    repo: str
    CodTrackerID: Optional[str]
    squad: List[str]
    gamertag: str
    headers: Optional[dict]

    def __init__(self, info: dict = None):

        if info is None:
            info = user_inputs

        self.file_name: str = info['file_name']
        self.USERNAME: str = info['username']
        self.PASSWORD: str = info['password']
        self.DRIVER_PATH: str = info['driverpath']
        self.repo: str = info['repo']
        self.CodTrackerID: str = info['codtrackerid']
        self.squad: List[str] = info['squad']
        self.gamertag: str = info['gamertag']
        self.headers: dict = info['headers']
