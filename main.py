import os
import pandas as pd
import numpy as np
from data_fetch import fetch_fred_series

FRED_API_KEY = os.getenv('FRED_API_KEY')
print(FRED_API_KEY)

# 1) Pull price
price_df = fetch_fred_series('PPIACO', FRED_API_KEY)
price_df.set_index('date', inplace=True)

print(price_df)