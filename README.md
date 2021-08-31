# Call-Of-Duty-Warzone-Analysis
[![Documentation Status](https://readthedocs.org/projects/call-of-duty-warzone-analysis/badge/?version=latest)](https://call-of-duty-warzone-analysis.readthedocs.io/en/latest/?badge=latest)

A package for analysis of _Call of Duty Warzone_ matches.

# Installation
    pip install warzone-analysis

# Usage

```python
import warzone
from warzone.call_of_duty import CallofDuty

user_input_dict = {'repo': 'location of saved data',
                   'gamertag': 'your Ganertag',
                   'squad': ['squadmate1', 'squadmate2', 'etc'],
                   'file_name': 'Match_Data.csv'}

cod = CallofDuty(user_input_dict=user_input_dict, squad_data=True, hacker_data=False, streamer_mode=False)
```