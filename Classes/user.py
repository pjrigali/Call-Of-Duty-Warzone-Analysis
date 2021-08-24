from typing import Optional, List
from dataclasses import dataclass
from credentials import user_inputs


@dataclass
class User:
    """
    Organizes the Users input data.

    Parameters
    ----------
    info : dict
        User input dict.

    Examples
    --------

    >>> from Classes.user import User
    >>> from credentials import user_inputs
    >>> user = User(info=user_input)

    """

    _file_name: str
    """File name of users data"""

    _repo: str
    """Directory location of data"""

    _gamertag: str
    """Users gamertag"""

    _squad: List[str]
    """List of gamertags"""

    # Need for Scraping
    headers: Optional[dict]
    """Headers from local machine"""

    CodTrackerID: Optional[str]
    """Cod Tracker ID for the user"""

    USERNAME: Optional[str]
    """Username for login to Cod Tracker"""

    PASSWORD: Optional[str]
    """Password for login to Cod Tracker"""

    DRIVER_PATH: Optional[str]
    """Driver path used for Selenium scraping"""

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
        """Returns the file name of the users data"""
        return self._file_name

    @property
    def repo(self) -> str:
        """Returns the directory location of the users data"""
        return self._repo

    @property
    def gamertag(self) -> str:
        """Returns the users gamertag"""
        return self._gamertag

    @property
    def squad(self) -> List[str]:
        """Returns the users squad gamertags as a list"""
        return self._squad
