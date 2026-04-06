"""
sessions.py
-----------
Utility for identifying regular US equity trading session bars.
"""

import pandas as pd


def is_regular_session(ts: pd.Series) -> pd.Series:
    """
    Vectorised check: returns True for timestamps within the regular US session.
    Regular session in UTC: 14:30 (inclusive) to 21:00 (exclusive).
    """
    ts     = pd.to_datetime(ts, utc=True)
    hour   = ts.dt.hour
    minute = ts.dt.minute

    after_open  = (hour > 14) | ((hour == 14) & (minute >= 30))
    before_close = hour < 21

    return after_open & before_close
