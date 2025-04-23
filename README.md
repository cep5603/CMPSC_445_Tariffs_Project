# CMPSC_445_Tariffs_Project

### Preliminary Plan

Commodity‑price forecasting with tariff rates

- Data sources:
    - FRED API for historical commodity prices (e.g. "PPIACO" series for steel, "PWAGSP" for agricultural)
    - USITC "Applied Rates of Duty" table (annual/quarterly CSV download by HS chapter)
- Preprocessing:
    - Pull time series via API + download tariff csv
    - Align on date (year‑quarter or month)
    - Fill small gaps with linear/interpolation
- Machine learning:
    - ARIMA/Prophet or LSTM on price series with tariff as exogenous regressor
- Web UI:
    - Choose commodity + forecast horizon, plot actual vs forecast
    - Maybe could also choose custom tariff rate
