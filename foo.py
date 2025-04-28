import requests
import pandas as pd

def get_steel_tariffs(start_year=2000, end_year=2024):
    """
    Fetch US MFN weighted average tariff rates for steel (HS2=72) by year
    using the WITS API.
    
    Returns a pandas Series with years as index and tariff rates as values.
    """
    tariffs = []
    
    for year in range(start_year, end_year + 1):
        # WITS API endpoint for tariff data
        url = (
            f"https://wits.worldbank.org/API/V1/SDMX/V21/datasource/tradestatstariff"
            f"/reporter/usa/year/{year}/partner/wld/product/72/indicator/AHS-WGHTD-AVRG"
        )
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Extract the tariff value from the XML response
            # The OBS_VALUE attribute contains the weighted average tariff
            if "OBS_VALUE" in response.text:
                # Simple string parsing to extract the value
                start_idx = response.text.find('OBS_VALUE="') + len('OBS_VALUE="')
                end_idx = response.text.find('"', start_idx)
                tariff_value = float(response.text[start_idx:end_idx])
                
                tariffs.append({
                    'year': year,
                    'tariff_rate': tariff_value
                })
                print(f"Year {year}: {tariff_value}%")
            else:
                print(f"No data for year {year}")
                
        except Exception as e:
            print(f"Error fetching data for {year}: {e}")
    
    # Convert to DataFrame and then to time series
    df = pd.DataFrame(tariffs)
    if df.empty:
        return pd.Series(name="tariff_rate")
    
    df['date'] = pd.to_datetime(df['year'], format="%Y")
    return df.set_index('date')['tariff_rate']

# Example usage
steel_tariffs = get_steel_tariffs(2000, 2024)
