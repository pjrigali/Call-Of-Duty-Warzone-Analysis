Glossary
========
.. meta::
   :description: This chapter serves as the glossary for this package.
   :keywords: Call of Duty, Warzone, Python, Data Science

Terms used in this documentation.

.. glossary::
    :sorted:

    :ref:`DocumentFilter <DocumentFilter>`
        Theoretically, the sound a parrot makes when four-thousand volts of electricity pass through it.

    *teamPlacement*
        An int value that represents a teams final place in a match.

    *placementPercent*
        The **teamPlacement** value converted to a float between 0 and 1.
        Changes depending on the teams final placement and what percent that represents out of the lobby.

    *matchId*
        A unique 16 digit number that represents a warzone match.

    *uno*
        A unique value that represents a single player. Activision creates these and gives one to each player.

    *map_choice*
        A DocumentFilter filter input. Either **'mp_e'** for Rebirth or **'mp_d'** for Verdansk.

    *mode_choice*
        A DocumentFilter filter input. Either **'solo'**, **'duo'**, **'trio'** or, **'quad'**.
        Will filter for only the specified mode.

    *col_lst*
        A list of columns to be analyzed from a given dataframe.

    *our_df*
        Data for the user and the users squad.

    *other_df*
        Data for other players that are not the user or the users squad.

    *whole*
        Data for all players within a users saved matches.
