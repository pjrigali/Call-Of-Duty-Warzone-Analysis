Intro
*****

.. meta::
   :description: Landing page for warzone-analysis.
   :keywords: Call of Duty, Warzone, Python, Data Science

`PYPI Page <https://pypi.org/project/warzone-analysis/2.4.1/>`_

.. code-block::

    pip install warzone-analysis

Usage
-----

.. code-block:: python

    import warzone
    from warzone import CallofDuty
    user_inputs = {'repo': 'local data directory',
                   'gamertag': 'your gamertag',
                   'squad': ['friend gamertag1', 'friend gamertag2', '... etc'],
                   'file_name': 'match_data.csv'}
    cod = CallofDuty(user_input_dict=user_inputs,
                     squad_data=True,
                     hacker_data=False,
                     streamer_mode=False)
This is the core class which gathers the users previous matches and holds all objects the user may need.

More Info
---------
`Github <https://github.com/pjrigali/Call-Of-Duty-Warzone-Analysis>`_

`Medium Posts <https://medium.com/@peterjrigali/warzone-package-part-1-b64d753e949c>`_
