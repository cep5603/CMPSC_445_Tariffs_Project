# CMPSC 445 – Tariffs Project

---

### Project Description

This project is a web application built to predict how much import values might change year-over-year for different countries and product types, specifically looking at the impact of tariffs. We fed historical trade data and average duty rates into a Random Forest model; this type of model is good at finding complex patterns. To make the predictions relevant, the model doesn't just look at current tariff levels, but focuses on changes: how much did the duty rate change this year? What was the change in import value last year (since past trends matter)? It also considers the duty rate level and change from the previous year. Of course, the specific country and product sector are key inputs too. Before training, we had to clean the data; this involved filling in missing tariff data using interpolation (guessing based on nearby years within the same group) and removing a few extreme outlier data points representing massive, unusual swings in import value to keep the model focused on more typical patterns. The final web app, built with Flask, lets users pick a country and sector, see the historical trends, plug in values for the key factors (or use a button to auto-fill for a "what if tariffs stay the same?" scenario), and get a prediction for the change in import value, shown in millions of USD.

---

### Project Significance

Quantitatively predicting the impact of import duties is notoriously difficult. This project's significance, then, comes from offering a practical tool that goes beyond just theory; it gives users a concrete prediction of how import values might react to tariff changes and recent history. Such an application, if further developed, could help businesses and policymakers anticipate the fallout from trade policy shifts, allowing for better planning. It's also meaningful because it demonstrates how techniques learned in class, like Random Forest regression, handling time-series features (like year-over-year changes and lags), dealing with outliers, and imputing missing data, can be applied to messy, real-world economic data.

---

### Instructions for Web Usage

Install the requirements:

`pip install -r requirements.txt`

(If not done so already, also install NodeJS.)

In a console window, run:

`python app.py`

to start the webserver.

Then, ctrl+click on the provided localhost link (default: `http://127.0.0.1:5001`) to open the interface.

Sample Output:

![image](https://github.com/user-attachments/assets/86915554-297e-42ad-8cbd-4503d969d297)


---

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
  - Saving the final merged_imports_duties.csv.

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

The web application contains the following functionalities with which the user may interact:

- Selection of reporting economy
- Selection of product sector
- Future tariff prediction
  - Four fields the user may additionally edit to predict the import values of future years:
     - Change in Average Duty Rate This Year (%)
     - Change in Average Duty Rate Last Year (%)
     - Average Duty Rate Last Year (%)
     - Change in Import Value Last Year (Millions USD)
     - *The user may also select an option to autofill the latter three with the values of the currently selected economy-product sector combination.*
 - Historical View
   - A line chart, made with Plotly, displaying the year-to-year import value and average duty rate of the selected economy-product sector combination

Naive Random Forest results:
RMSE: 80797.07
R²: 0.98

Delta/lagged Random Forest results:
Train: R²=0.9323, RMSE=24222.95
Test:  R²=0.7535, RMSE=48768.48

(As can be seen, the model still overfits.)

---

### Data Collection

Data was collected from `https://stats.wto.org/` after creating a WTO account. All reporting economies and the years 1988–2024 were selected. Datasets for the following two economic indicators were downloaded:

- Merchandise imports by product group – annual (Million US dollar)
  - *This is used as a target variable*
  - Data quantity: 102136 data rows
- MFN - Simple average duty by product groups (Percent)
  - *This is used as an explanatory variable*
  - Data quantity: 49523 data rows
 
Each row also includes many metadata not directly used by this project, such as the partner economy (only reporter economies were considered).

The final dataframe used for model training is structured as below.

(The row order in this sample is randomized to show data diversity.)

| Year | Product/Sector                                   | Reporting Economy         | ImportValue   | AverageDutyRate |
| :--- | :----------------------------------------------- | :------------------------ | :------------ | :-------------- |
| 2017 | Fuels and mining products                        | Bosnia and Herzegovina    | 1909.826295   |                 |
| 2011 | Textiles                                         | Zimbabwe                  | 87.926265     | 10.36           |
| 2019 | Manufactures                                     | Hungary                   | 96326.13703   |                 |
| 2022 | Pharmaceuticals                                  | Portugal                  | 3967.788733   |                 |
| 2019 | Food                                             | Seychelles                | 273.456756    | 11.56111        |
| 2008 | Agricultural products                            | Saudi Arabia, Kingdom of  | 12801.95974   | 4.435484        |
| 2008 | Office and telecom equipment                     | Syrian Arab Republic      | 194.241765    |                 |
| 1988 | Office and telecom equipment                     | Sri Lanka                 | 73.387728     |                 |
| 2023 | Chemicals                                        | Chinese Taipei            | 33103.372     | 2.768941        |
| 2009 | Manufactures                                     | Honduras                  | 4814.202615   | 6.505           |
| 2011 | Electronic data processing and office equipment  | Saint Lucia               | 12.083896     | 10.21           |
| 1994 | Agricultural products                            | France                    | 30824.63894   |                 |
| 2006 | Food                                             | European Union            | 219987.832    | 21.35452        |
| 2021 | Clothing                                         | Greenland                 | 27.48         |                 |
| 2008 | Manufactures                                     | Solomon Islands           | 151.917941    | 9.871749        |

---

### Data Processing

A number of preprocessing steps were taken to prepare the data for training. Firstly, it was necessary to combine the two datasets into one, with all the relevant features. There was an immediate challenge with this: the specific product sectors used by each table are not standardized across the two. The tariff data uses the WTO's own standardized 22-product categorization scheme, while the import data uses a different, SITC3 product classification, which had 18 unique values in this case. To address this, a mapping module was created (`mapper.py`) that attempts to merge the two product/sector schemes with as little loss in information and intelligibility as possible.

Although data was selected to be downloaded starting from 1988, tariff data were only available for 2005 and beyond, so import data for years before that were dropped.

The data also included information on supernational entities like `World` and `European Union`. The values in these are very large and encompass many countries already included in the dataset, so entries with reporters such as these were dropped.

One other step was data imputation: many tariff values (average duty rates) were missing values intermittently, so imputation was performed to fill these in with the smoothed average values of surrounding years.

Outlier removal and normalization of import values according to the reporter were also attempted as preprocessing steps, but they did not improve performance (at least, as implemented).

---

### Model Development

Initially, linear regression was used, but this performed very poorly, so random forest was tested next instead. This was able to perform much better, initially achieving about 94% accuracy but a poor RMSE of about 25,000. Through data preprocessing, the accuracy was reduced but the RMSE improved, creating a more well-rounded model.

The random forest model was tested with and without lagged features. Without lagged features (especially `Lag1_Delta_TargetValue`), the model struggles significantly more to predict even the non-extreme changes accurately. The lagged features were determined to be important for achieving both high R² and low RMSE on the outlier-removed data.

Top 20 Feature Importances on final model:

![Feature_Importances_Updated](https://github.com/user-attachments/assets/7491ef0c-d8ff-4c7e-85fe-59de03703d65)


---

### Discussion and Conclusion

In both preliminary and final models, the R² is quite high, but RMSE is poor. This indicates that the model is good at predicting the vast majority of year-to-year import value deltas, but struggles with outliers.

There was overall a tradeoff between maximizing R² and minimizing RMSE. It was very easy to get a good score on only one of these, but extremely difficult to do so for both. For instance, during development, one model was developed with an R² of 94%, but an RMSE of over 22,000. This suggested that the model was excellent at predicting the vast majority of cases where the year-to-year change in import values was small, but made very large errors when predicting the magnitude (but not direction) of multi-billion dollar swings.
