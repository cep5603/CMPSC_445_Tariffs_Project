# CMPSC_445_Tariffs_Project

### Project Description

This project is a web application built to predict how much import values might change year-over-year for different countries and product types, specifically looking at the impact of tariffs. We fed historical trade data and average duty rates into a Random Forest model; this type of model is good at finding complex patterns. To make the predictions relevant, the model doesn't just look at current tariff levels, but focuses on changes – how much did the duty rate change this year? What was the change in import value last year (since past trends matter)? It also considers the duty rate level and change from the previous year. Of course, the specific country and product sector are key inputs too. Before training, we had to clean the data; this involved filling in missing tariff data using interpolation (guessing based on nearby years within the same group) and removing a few extreme outlier data points representing massive, unusual swings in import value to keep the model focused on more typical patterns. The final web app, built with Flask, lets users pick a country and sector, see the historical trends, plug in values for the key factors (or use a button to auto-fill for a "what if tariffs stay the same?" scenario), and get a prediction for the change in import value, shown in millions of USD.

### Project Significance

Quantitatively predicting the impact of import duties is notoriously difficult. This project's significance, then, comes from offering a practical tool that goes beyond just theory; it gives users a concrete prediction of how import values might react to tariff changes and recent history. Such an application, if further developed, could help businesses and policymakers anticipate the fallout from trade policy shifts, allowing for better planning. It's also meaningful because it demonstrates how techniques learned in class, like Random Forest regression, handling time-series features (like year-over-year changes and lags), dealing with outliers, and imputing missing data, can be applied to messy, real-world economic data.

### Instructions for Web Usage

Install the requirements:

`pip install -r requirements.txt`

(If node done so already, also install NodeJS.)

In a console window, run:

`python app.py`

to start the webserver.

Then, ctrl+click on the provided localhost link (default: `http://127.0.0.1:5001`) to open the interface.

### Code Structure

Core Application Logic:

- app.py: The main Flask web application file. It handles:
  - Loading the trained model (.joblib file).
  - Defining web routes (e.g., for the homepage /, prediction /predict, historical data /plot_history).
  - Processing user input from the web form.
  - Calling the model's prediction function.
  - Fetching data for visualizations.
  - Rendering HTML templates (index.html) to display the UI and results.

Model Training & Analysis:

- train_delta_model.py: Contains the primary machine learning workflow. This includes:
  - Loading the merged data.
  - Performing feature engineering (calculating deltas, lags).
  - Handling missing values (interpolation).
  - Implementing outlier removal.
  - Defining the preprocessing steps (scaling, encoding).
  - Defining and training the RandomForestRegressor model pipeline.
  - Evaluating the model (R², RMSE) on train/test sets.
  - Saving the trained model pipeline and the list of features used.

Data Processing & Utilities:

- mapper.py: Responsible for the initial data merging process, including:
  - Loading the raw import and duty CSV files.
  - Mapping the different product/sector categories between the two datasets.
  - Aggregating duty rates.
  - Merging the two datasets based on Year, Reporting Economy, and Product/Sector.
  = Saving the final merged_imports_duties.csv.

- data_utils.py: Contains helper functions used by other scripts:
  - `filter_economies()`
  - `verify_interpolation()`

- Raw Data Files:
  - `Merchandise imports by product group.csv`
    - Input CSV containing import value data.
  - `Simple average duty by product group.csv`
    - Input CSV containing duty rate data.

Generated Outputs/Visualizations:

- all_hs2_tariffs.png: A plot generated during initial data exploration or analysis of tariff rates.
- feature_importance_top20.png: A plot visualizing the most important features identified by one of the models during development.

Key Directories:

- data/: Contains the processed, merged data file (merged_imports_duties.csv) used directly by the training script and Flask app.
- fred_cache/: Stores cached data downloaded from the FRED database (a couple of raw PPI series).
- model/: Stores the serialized machine learning model (tariff_rf_pipeline.joblib) and the list of features it expects (model_features.joblib). This allows the Flask app to load and use the trained model without retraining.
- templates/: Flask directory containing the HTML file(s) (index.html) that define the structure and content of the web pages.
- yearly_us_data/: Contains raw, year-specific tariff data (original Excel files from USITC) used in earlier iterations of this project

### Functionalities and Test Results

### Data Collection

Data was collected from `https://stats.wto.org/` after creating a WTO account. All reporting economies and the years 1988–2024 were selected. The following two indicators were attained:

- Merchandise imports by product group – annual (Million US dollar)
 - *This is used as a target variable*
 - Data quantity: 102136 data rows
- MFN - Simple average duty by product groups (Percent)
  - *This is used as an explanatory variable*
  - Data quantity: 49523 data rows
 
Each row also includes many metadata not directly used by this project, such as the partner economy (only reporter economies were considered).

### Data Processing

A number of preprocessing steps were taken to prepare the data for training. Firstly, it was necessary to combine the two datasets into one, with all the relevant features. There was an immediate challenge with this: the proucts sectors are not standardized across the two. The tariff data uses the WTO's own standardized 22-product categorization scheme, while the import data uses a different, SITC3 product classification, which had 18 unique values in this case. To address this, a mapping module was created (`mapper.py`) that attempts to merge the two product/sector schemes with as little loss in information and intelligibility as possible.

Although data was selected to be downloaded starting from 1988, tariff data were only available for 2005 and beyond, so import data for years before that were dropped.

The data also included information on supernational entities like `World` and `European Union`. The values in these are very large and encompass many countries already included in the dataset, so entries with reporters such as these were dropped.

One other step was data imputation: many tariff values (average duty rates) were missing values intermittently, so imputation was performed to fill these in with the smoothed average values of surrounding years.

Outlier removal and normalization of import values according to the reporter were also attempted as preprocessing steps, but they did not improve performance (at least, as implemented).

### Model Development

Initially, linear regression was used, but this performed very poorly, so random forest was tested next instead. This was able to perform much better, initially achieving about 94% accuracy but a poor RMSE of about 25,000. Through data preprocessing, the accuracy was reduced but the RMSE improved, creating a more even model.

### Discussion and Conclusion

In both preliminary and final models, the R^2 is quite high, but RMSE is poor. This indicates that the model is good at predicting the vast majority of year-to-year import value deltas, but struggles with outliers.
