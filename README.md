# Call of Duty: *Warzone Analysis*
[![Documentation Status](https://readthedocs.org/projects/call-of-duty-warzone-analysis/badge/?version=latest)](https://call-of-duty-warzone-analysis.readthedocs.io/en/latest/?badge=latest)

A package for analysis of _Call of Duty Warzone_ matches.

## Description

The primary goal of this package is to explore a player (or squads) performance. 
The secondary goal is help the user understand the conditions that affect how they preform.

## Getting Started

### Dependencies

* numpy
* matplotlib
* pandas
* requests
* selenium
* scipy
* yellowbrick
* statsmodels
* scikit-learn

### Installing

The package can be accessed via pip install.
[Pypi Documentation](https://pypi.org/project/warzone-analysis/)

    pip install warzone-analysis

### Executing program

[Read the Docs](https://call-of-duty-warzone-analysis.readthedocs.io/en/latest/intro.html)

It is recommended that you have a pre-existing dataset to examine.
If you are starting from scratch I would recommend getting your most recent matches from the following endpoint.

_https://api.tracker.gg/api/v1/warzone/matches/:platform:/:username:/aggregated?localOffset=300_

Platform is one of the following:
* battlenet
* atvi
* psn
* xbl

Username can be found by accessing your profile on Codtracker. 
Need the number as well. __Replace the # with %23.__

Request the Json. You really only need the Match Ids. 
Once you have a list of Match Ids you can connect to the Call of Duty API. 

_For more info see: [Scraping and Api Connection](https://medium.com/@peterjrigali/warzone-package-part-1-b64d753e949c)_


```python
import warzone
from warzone.call_of_duty import CallofDuty

user_input_dict = {'repo': 'location of saved data',
                   'gamertag': 'your Ganertag',
                   'squad': ['squadmate1', 'squadmate2', 'etc'],
                   'file_name': 'Match_Data.csv'}

cod = CallofDuty(user_input_dict=user_input_dict, squad_data=True, hacker_data=False, streamer_mode=False)
```

**Below are some write-ups detailing and utilizing the package.**

[Plot Examples](https://medium.com/@peterjrigali/warzone-package-part-3-c1cfa2be46bc)
_Examples for using the built-in Line, Scatter, Histogram and Table Classes._

[Squad Class Example](https://medium.com/@peterjrigali/warzone-package-part-2-3ff94902f355)
_Exploring the Squad Class built inside CallofDuty Class._

[Scraping and Api Connection](https://medium.com/@peterjrigali/warzone-package-part-1-b64d753e949c)
_Outlines process for grabbing your past matches and connecting to the Call of Duty public endpoints._

[More Hackers Now?](https://medium.com/@peterjrigali/more-hackers-now-51c7cbe0ac87)
_Exploring tracking hackers in your lobby and how these numbers have changed over time._

[Engagement Based Matchmaking in Warzone](https://medium.com/@peterjrigali/engagement-based-matchmaking-in-warzone-part-1-48b1ef72ada0)
_Exploring the evidence for Engagement Based Matchmaking_

[Detect Hackers in your Warzone Lobby](https://medium.com/@peterjrigali/how-to-tell-if-hackers-are-in-your-warzone-lobby-part-1-393360c38104)
_Do lobbies die out quicker when hackers are present?_
