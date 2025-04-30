import pandas as pd

# Assuming df_imports and df_duties are your DataFrames
df_imports = pd.read_csv('Merchandise imports by product group.csv', encoding='latin-1')
df_duties = pd.read_csv('Simple average duty by product group.csv', encoding='latin-1')

unique_imports = df_imports['Product/Sector'].unique()
unique_duties = df_duties['Product/Sector'].unique()

print("Import Categories:", len(unique_imports))
print(sorted(unique_imports)) # Sort for easier comparison
print("\nDuty Categories:", len(unique_duties))
print(sorted(unique_duties))