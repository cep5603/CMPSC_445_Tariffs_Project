import os
import pandas as pd
import numpy as np
import requests
from data_fetch import fetch_fred_series

FRED_API_KEY = os.getenv('FRED_API_KEY')
print(FRED_API_KEY)

series = {
    'ppi-all': 'PPIACO',
    'ppi-steel': 'WPU10170502',
    'ppi-soybeans': 'WPU01830131'
}

def main():
    print("hll")
    response = requests.get("https://wits.worldbank.org/API/V1/SDMX/V21/rest/data/df_wits_tradestats_tariff/A.usa.wld.fuels.AHS-SMPL-AVRG?startPeriod=2000&endPeriod=2002")
    response.raise_for_status()
    print(response.text)


    # 1) Pull price
    #price_df = fetch_fred_series(series['steel'], FRED_API_KEY)
    #price_df.set_index('date', inplace=True)
    #print(price_df)

if __name__ == '__main__':
    main()