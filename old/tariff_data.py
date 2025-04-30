import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional

from tariff_plots import plot_top_tariff_rates

def load_annual_tariffs_hs2_list(hs2_codes: Optional[List[str]] = None, data_glob: str = 'yearly_us_data/tariff_data_*/tariff_database_*.xlsx', cache_dir: str = '.', outlier_threshold: float = 1000.0) -> pd.DataFrame:
    """
    Reads all HTS-8 Excel files matching data_glob,
    filters to HS2 codes (defaults to '10' through '97'),
    drops any mfn_ad_val_rate > outlier_threshold,
    computes the annual mean of mfn_ad_val_rate per HS2,
    pivots to wide form (one column per HS2), and caches.
    """
    # default to HS2 = '10','11',…,'97'
    if hs2_codes is None:
        codes_sorted = [f'{i:02d}' for i in range(10, 98)]
        cache_fn = os.path.join(cache_dir, 'annual_tariffs_10-97.pkl')
    else:
        codes_sorted = sorted(str(c) for c in hs2_codes)
        cache_fn = os.path.join(cache_dir, f"annual_tariffs_{'_'.join(codes_sorted)}.pkl")

    # return cached if exists
    if os.path.exists(cache_fn):
        return pd.read_pickle(cache_fn)

    # 1) Load all yearly Excels & collect relevant columns
    records = []
    for fn in glob.glob(data_glob):
        print(f'Reading: {fn}')
        df = pd.read_excel(fn, dtype={'hts8': str})
        if 'Year' not in df.columns:
            m = re.search(r'tariff_database_(\d{4})', fn)
            if not m:
                raise ValueError(f'Cannot infer Year from {fn}')
            df['Year'] = int(m.group(1))
        df['HS2'] = df['hts8'].str[:2]
        records.append(df[['Year', 'HS2', 'mfn_ad_val_rate']])

    all_df = pd.concat(records, ignore_index=True)

    # 2) Filter to requested HS2 codes and remove outliers
    all_df = all_df[all_df['HS2'].isin(codes_sorted) & (all_df['mfn_ad_val_rate'] <= outlier_threshold)]

    # 3) Group by Year & HS2, compute mean, then pivot to wide form
    agg = (all_df.groupby(['Year', 'HS2'])['mfn_ad_val_rate'].mean().reset_index())
    pivot = agg.pivot(index='Year', columns='HS2', values='mfn_ad_val_rate')

    # 4) Ensure every HS2 column is present
    pivot = pivot.reindex(columns=codes_sorted)

    # 5) Rename columns and set a datetime index
    pivot.columns = [f'tariff_{c}' for c in pivot.columns]
    pivot.index = pd.to_datetime(pivot.index.astype(str), format='%Y')

    # 6) Cache and return
    pivot.to_pickle(cache_fn)
    return pivot


pd.set_option('display.max_rows', None)

df_all = load_annual_tariffs_hs2_list()
#print(df_all)

fig, ax = plot_top_tariff_rates(df_all, top_n=10, by='mean', title='Top 10 HS2 Tariffs by Average Rate')
fig.savefig('top10_hs2_tariffs.png', dpi=300)