from fredapi import Fred
import pandas as pd

def fetch_fred_series(series_id, api_key, start_date=None, end_date=None):
    """
    Fetches a FRED time series as a DataFrame.

    series_id:    the FRED series code (e.g. 'PPIACO')
    api_key:      your FRED API key
    start_date:   'YYYY-MM-DD' or None
    end_date:     'YYYY-MM-DD' or None

    Returns a DataFrame with columns ['date', 'value'].
    """

    fred = Fred(api_key=api_key)
    raw = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    df = raw.reset_index()
    df.columns = ['date', 'value']

    # Ensure correct dtypes
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df
