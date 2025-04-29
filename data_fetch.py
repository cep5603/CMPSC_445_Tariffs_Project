import os
import pandas as pd
from fredapi import Fred

def fetch_fred_series(series_id, api_key, start_date=None, end_date=None):
    """
    Fetches a FRED time series as a DataFrame, using a CSV cache.

    series_id:  the FRED series code (e.g. 'PPIACO')
    api_key:    your FRED API key
    start_date: 'YYYY-MM-DD' or None
    end_date:   'YYYY-MM-DD' or None

    Returns a DataFrame with columns ['date', 'value'].
    """
    cache_dir = 'fred_cache'
    os.makedirs(cache_dir, exist_ok=True)

    # Build cache filename based on series and date range
    start = start_date or 'start'
    end = end_date or 'end'
    fname = f'{series_id}_{start}_{end}.csv'
    cache_path = os.path.join(cache_dir, fname)

    # If cached file exists, load and return it
    if os.path.exists(cache_path):
        print('Getting cached value:')
        df = pd.read_csv(cache_path, parse_dates=['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df

    # Otherwise fetch from FRED
    fred = Fred(api_key=api_key)
    raw = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    df = raw.reset_index()
    df.columns = ['date', 'value']
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # Cache for next time
    df.to_csv(cache_path, index=False)
    return df
