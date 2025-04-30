# Mapping from Duty Categories (Keys) to Import Categories (Values)

duty_to_import_map = {
    # --- Agriculture / Food Group ---
    'Animal products': 'Food', # Aggregate into Food
    'Beverages and tobacco': 'Food', # Aggregate into Food
    'Cereals and preparations': 'Food', # Aggregate into Food
    'Coffee, tea': 'Food', # Aggregate into Food
    'Dairy products': 'Food', # Aggregate into Food
    'Fish and fish products': 'Food', # Aggregate into Food
    'Fruits, vegetables, plants': 'Food', # Aggregate into Food
    'Oilseeds, fats and oils': 'Food', # Aggregate into Food
    'Sugars and confectionery': 'Food', # Aggregate into Food
    'Other agricultural products': 'Agricultural products', # Specific match

    # --- Textiles / Clothing Group ---
    'Clothing': 'Clothing', # Exact match
    'Cotton': 'Textiles', # Raw material for Textiles
    'Leather, footwear, etc': 'Clothing', # Closest fit, though imperfect. Could also be 'Manufactures'.
    'Textiles': 'Textiles', # Exact match

    # --- Chemicals / Pharma Group ---
    'Chemicals': 'Chemicals', # Exact match
    # Note: 'Pharmaceuticals' exists in Imports but not Duties. Chemicals is the closest duty category.

    # --- Minerals / Metals / Fuels Group ---
    'Minerals and metals': 'Iron and steel', # Mapping broader duty category to the most prominent related import category. Could also map to 'Fuels and mining products' or 'Manufactures'. This is an approximation.
    'Petroleum': 'Fuels', # Closest specific match

    # --- Machinery / Electronics / Transport Group ---
    'Electrical machinery': 'Electronic data processing and office equipment', # Best fit among specific electronic import categories
    'Non-electrical machinery': 'Machinery and transport equipment', # Fits into the broader machinery category
    'Transport equipment': 'Transport equipment', # Exact match
    # Note: Imports have more specific electronic/telecom categories not directly matched in Duties.

    # --- Other Manufactures ---
    'Manufactures n.e.s.': 'Manufactures', # Exact match for the general category
    'Wood, paper, etc': 'Manufactures', # Fits into general manufactures

    # --- Categories to potentially exclude or handle separately ---
    # 'Total merchandise' from Imports is an aggregate, likely not useful for mapping *from* duties.
}

import pandas as pd

df_imports = pd.read_csv('Merchandise imports by product group.csv', encoding='latin-1')
df_duties = pd.read_csv('Simple average duty by product group.csv', encoding='latin-1')

# Rename in Imports DataFrame
if 'Value' in df_imports.columns:
    df_imports = df_imports.rename(columns={'Value': 'ImportValue'})
    print("Renamed 'Value' to 'ImportValue' in df_imports.")
else:
    print("Warning: 'Value' column not found in df_imports.")

# Rename in Duties DataFrame (BEFORE cleaning/aggregation)
if 'Value' in df_duties.columns:
    df_duties = df_duties.rename(columns={'Value': 'AverageDutyRate'})
    print("Renamed 'Value' to 'AverageDutyRate' in df_duties.")
    original_duty_value_col = 'AverageDutyRate'
else:
    print("Warning: 'Value' column not found in df_duties.")

# Standardize duty categories
df_duties['Standardized_Sector'] = df_duties['Product/Sector'].map(duty_to_import_map)

# Aggregate duty rates
df_duties_clean = df_duties[['Year', 'Standardized_Sector', original_duty_value_col]].dropna()

print('\nChecking for duplicates in duties BEFORE aggregation:')
print(df_duties_clean.duplicated(subset=['Year', 'Standardized_Sector']).sum())

# Group by the keys we will merge on and calculate the mean duty rate
# Use the RENAMED duty column name for aggregation
df_duties_agg = df_duties_clean.groupby(
    ['Year', 'Standardized_Sector']
)[original_duty_value_col].mean().reset_index()
# The aggregated column now has the name stored in original_duty_value_col ('AverageDutyRate')

print("\nShape of aggregated duties:", df_duties_agg.shape)
print("Checking for duplicates in duties AFTER aggregation:")
print(df_duties_agg.duplicated(subset=['Year', 'Standardized_Sector']).sum()) # Should be 0

# Check imports granularity/dupes
print("\nShape of imports:", df_imports.shape)
print("Import columns:", df_imports.columns)
print("Checking for duplicates in imports on merge keys:")
print(df_imports.duplicated(subset=['Year', 'Product/Sector']).sum())


# Perform merge using AGGREGATED duties
print("\nMerging...")
df_merged = pd.merge(df_imports, df_duties_agg, left_on=['Year', 'Product/Sector'], right_on=['Year', 'Standardized_Sector'], how='left')

print("\nFinal merged shape:", df_merged.shape)
print("Columns in merged DataFrame:", df_merged.columns.tolist())

necessary_cols = ['Year', 'Product/Sector', 'Reporting Economy', 'ImportValue', 'AverageDutyRate']
df_merged_small = df_merged[necessary_cols]
print("Shape after selecting columns:", df_merged_small.shape)

SAVE_NAME = 'merged_imports_duties.csv'
df_merged_small.to_csv(SAVE_NAME, index=False)
print(f'Saved to {SAVE_NAME}')