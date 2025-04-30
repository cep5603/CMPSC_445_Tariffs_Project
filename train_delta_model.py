# train_delta_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
# --- Model Change ---
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor # Import Random Forest
# --- End Model Change ---
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict, Any, List # Added List
# --- Plotting Imports ---
import matplotlib.pyplot as plt
import seaborn as sns
# --- End Plotting Imports ---

def filter_economies(df: pd.DataFrame, exclude_list: list = ['World', 'European Union']) -> pd.DataFrame:
    """
    Removes rows corresponding to specified reporting economies.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a 'Reporting Economy' column.
    exclude_list : list, optional
        List of economy names to exclude
        (default: ['World', 'European Union']).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    initial_rows = len(df)
    if not exclude_list:
        print("No economies specified for exclusion.")
        return df

    print(f"\nFiltering out reporting economies: {exclude_list}")
    mask = df['Reporting Economy'].isin(exclude_list)
    df_filtered = df[~mask].copy() # Use .copy() to avoid SettingWithCopyWarning later
    removed_rows = initial_rows - len(df_filtered)
    print(f"Removed {removed_rows} rows corresponding to excluded economies.")
    print(f"Shape after filtering: {df_filtered.shape}")
    return df_filtered

def train_import_delta_model(
    csv_path: str,
    economies_to_exclude: list = ['World', 'European Union'],
    test_size: float = 0.2,
    random_state: int = 42,
    use_lags: bool = True,
    use_one_hot: bool = True,
    scale_numeric: bool = True,
    plot_target_dist: bool = True, # Option to plot
    rf_n_estimators: int = 100, # RF hyperparameter
    rf_max_depth: int = None,   # RF hyperparameter
    rf_n_jobs: int = -1         # RF hyperparameter (-1 uses all cores)
) -> Tuple[Pipeline, Dict[str, Any], List[str], pd.DataFrame]: # Adjusted return type hint
    """
    Trains a RandomForestRegressor model to predict the change in ImportValue
    based on the change in AverageDutyRate, controlling for economy/sector.
    Optionally plots the target variable distribution.

    Parameters
    ----------
    csv_path : str
        Path to the 'merged_imports_duties.csv' file.
    test_size : float, optional
        Fraction of data for the test set.
    random_state : int, optional
        Random seed for splitting data.
    use_lags : bool, optional
        Whether to include lagged delta features.
    use_one_hot : bool, optional
        Whether to one-hot encode categorical features.
    scale_numeric : bool, optional
        Whether to scale numeric features.
    plot_target_dist : bool, optional
        If True, plots the distribution of Delta_ImportValue.
    rf_n_estimators : int, optional
        Number of trees in the random forest.
    rf_max_depth : int or None, optional
        Maximum depth of the trees.
    rf_n_jobs : int, optional
        Number of jobs to run in parallel for RF fitting (-1 means using all processors).


    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        The fitted preprocessing and model pipeline.
    metrics : dict
        Dictionary containing 'RMSE' and 'R2_Score' on the test set.
    feature_names_out : list
        List of feature names after transformations.
    debug_df : pd.DataFrame
         DataFrame containing calculated deltas and lags for inspection.
    """
    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded data: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return None, None, None, None

    required_cols = ['Year', 'Product/Sector', 'Reporting Economy', 'ImportValue', 'AverageDutyRate']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain columns: {required_cols}")
        return None, None, None, None

    # --- Add Filtering Step ---
    df = filter_economies(df, exclude_list=economies_to_exclude)
    if df.empty:
        print("DataFrame is empty after filtering economies. Stopping.")
        return None, None, None, None
    # --- End Filtering Step ---

    df['Year'] = pd.to_numeric(df['Year'])
    df = df.sort_values(by=['Reporting Economy', 'Product/Sector', 'Year'])

    # 2. Calculate Deltas
    group_cols = ['Reporting Economy', 'Product/Sector']
    df['Delta_ImportValue'] = df.groupby(group_cols)['ImportValue'].diff()
    df['Delta_AverageDutyRate'] = df.groupby(group_cols)['AverageDutyRate'].diff()

    # 3. Feature Engineering (Optional Lags)
    numeric_features = ['Delta_AverageDutyRate']
    if use_lags:
        df['Lag1_Delta_AverageDutyRate'] = df.groupby(group_cols)['Delta_AverageDutyRate'].shift(1)
        df['Lag1_Delta_ImportValue'] = df.groupby(group_cols)['Delta_ImportValue'].shift(1)
        df['Lag1_AverageDutyRate'] = df.groupby(group_cols)['AverageDutyRate'].shift(1)
        numeric_features.extend([
            'Lag1_Delta_AverageDutyRate',
            'Lag1_Delta_ImportValue',
            'Lag1_AverageDutyRate'
        ])
        print("Using lagged features.")

    # 4. Prepare Data for Modeling
    target = 'Delta_ImportValue'
    df_model = df.dropna(subset=[target] + numeric_features).copy()
    print(f"Shape after dropping NaNs from diff/shift: {df_model.shape}")

    if df_model.empty:
        print("Error: No data remaining after calculating differences/lags.")
        return None, None, None, None

    # --- Plot Target Distribution ---
    if plot_target_dist:
        plt.figure(figsize=(10, 6))
        sns.histplot(df_model[target], kde=True, bins=50)
        plt.title(f'Distribution of {target}')
        plt.xlabel('Change in Import Value')
        plt.ylabel('Frequency')
        # Consider adding log scale or trimming outliers if distribution is highly skewed
        plt.yscale('log')
        lower_quantile = df_model[target].quantile(0.01)
        upper_quantile = df_model[target].quantile(0.99)
        sns.histplot(df_model[target].clip(lower_quantile, upper_quantile), kde=True, bins=50)
        plt.tight_layout()
        plt.show()
    # --- End Plotting ---

    # Define features
    categorical_features = ['Reporting Economy', 'Product/Sector']
    if not use_one_hot:
        categorical_features = []

    features = numeric_features + categorical_features
    X = df_model[features]
    y = df_model[target]

    # 5. Preprocessing Pipeline
    transformers = []
    if scale_numeric and numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
        print("Applying StandardScaler to numeric features.")
    if use_one_hot and categorical_features:
        # Sparse matrix is default and usually fine for RF, but set to False if needed elsewhere
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features))
        print("Applying OneHotEncoder to categorical features.")

    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    else:
        print("Warning: No scaling or encoding applied.")
        preprocessor = 'passthrough'

    # 6. Model Pipeline (Using RandomForestRegressor)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # --- Model Change ---
        ('regressor', RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=random_state,
            n_jobs=rf_n_jobs,
            oob_score=True # Useful for quick performance check without test set
        ))
        # --- End Model Change ---
    ])

    # 7. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 8. Train Model
    print("Training RandomForest model...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")
    # Access OOB score if calculated
    try:
        oob = pipeline.named_steps['regressor'].oob_score_
        print(f"Model OOB Score: {oob:.4f}")
    except AttributeError:
        print("OOB score not available.")


    # Get feature names after potential transformation
    try:
        # Note: get_feature_names_out might behave differently with sparse matrices from OHE
        # depending on sklearn version. May need adjustments if errors occur.
        feature_names_out = pipeline.named_steps['preprocessor'].get_feature_names_out()
    except Exception as e:
        print(f"Could not get transformed feature names automatically: {e}. Using original.")
        feature_names_out = features # Fallback

    # 9. Evaluate
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {'RMSE': rmse, 'R2_Score': r2}
    print(f"Evaluation Metrics: {metrics}")

    return pipeline, metrics, list(feature_names_out), df_model # Return list


# --- Example Usage ---
if __name__ == '__main__':
    FILE_PATH = 'merged_imports_duties.csv'

    print("--- Training RandomForest Model ---")
    pipeline_rf, metrics_rf, features_out_rf, debug_df_rf = train_import_delta_model(
        FILE_PATH,
        economies_to_exclude=['World'],
        use_lags=True,
        use_one_hot=True,
        scale_numeric=True,
        plot_target_dist=True, # Enable plotting
        rf_n_estimators=100,   # Example hyperparameter
        rf_n_jobs=-1
    )

    if pipeline_rf:
        print("\nRandomForest model trained successfully.")

        # --- Feature Importance for RandomForest ---
        try:
            # Access feature importances from the RF step
            importances = pipeline_rf.named_steps['regressor'].feature_importances_
            # Create a Series for easier handling
            importance_series = pd.Series(importances, index=features_out_rf)
            # Sort and print top N
            top_n = 20
            print(f"\nTop {top_n} Feature Importances (Random Forest):")
            print(importance_series.sort_values(ascending=False).head(top_n))

            # Optional: Plot feature importances
            plt.figure(figsize=(10, max(5, top_n * 0.3)))
            importance_series.sort_values(ascending=False).head(top_n).plot(kind='barh')
            plt.title(f'Top {top_n} Feature Importances (Random Forest)')
            plt.xlabel('Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"\nCould not retrieve or plot feature importances: {e}")

        # print("\nDebug DataFrame head:")
        # print(debug_df_rf.head())
