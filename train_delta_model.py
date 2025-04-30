import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict, Any, List, Literal, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from data_utils import *

def train_import_delta_model(
    csv_path: str,
    start_year: int = 2005,
    target_transform: Literal['none', 'log_diff', 'pct_change'] = 'none',
    outlier_percentile_threshold: Optional[float] = None,
    economies_to_exclude: list = ['World', 'United States of America', 'China'],#['World'],
    test_size: float = 0.2,
    random_state: int = 2,
    use_lags: bool = True,
    use_one_hot: bool = True,
    scale_numeric_features: bool = True,
    plot_target_dist: bool = True,
    rf_n_estimators: int = 100,
    rf_max_depth: int = None,
    rf_n_jobs: int = -1
) -> Tuple[Pipeline, Dict[str, Dict[str, float]], List[str], pd.DataFrame]:
    # Load from CSV
    df = pd.read_csv(csv_path)
    print(f'Loaded data of shape: {df.shape}')

    if start_year is not None:
        initial_rows_year = len(df)
        #print(f"\nFiltering data to include years >= {start_year}...")
        df = df[df['Year'] >= start_year].copy() # Use .copy()
        removed_rows_year = initial_rows_year - len(df)
        #print(f"Removed {removed_rows_year} rows from years before {start_year}.")
        #print(f"Shape after year filter: {df.shape}")
        if df.empty:
            #print(f"Error: No data remaining after filtering for year >= {start_year}.")
            return None, None, None, None

    # Filter economies
    if economies_to_exclude:
        df = filter_economies(df, exclude_list=economies_to_exclude)
        if df.empty:
            return None, None, None, None

    # Ensure types and sort
    df['Year'] = pd.to_numeric(df['Year'])
    df = df.sort_values(by=['Reporting Economy', 'Product/Sector', 'Year'])

    duty_col = 'AverageDutyRate'
    initial_nans = df[duty_col].isnull().sum()
    do_interpolation_check = False
    df_original_for_verify = df.copy() if do_interpolation_check and initial_nans > 0 else None

    if initial_nans > 0:
        #print(f"\nFound {initial_nans} missing values in '{duty_col}'. Applying group-wise linear interpolation...")
        # Apply interpolation within each group
        df[duty_col] = df.groupby(['Reporting Economy', 'Product/Sector'])[duty_col].transform(lambda x: x.interpolate(method='linear', limit_direction='both', limit_area=None))
        # limit_direction='both' fills forward then backward
        # limit_area=None ensures it fills all consecutive NaNs if possible

        remaining_nans = df[duty_col].isnull().sum()
        filled_nans = initial_nans - remaining_nans
        #print(f"Interpolated {filled_nans} values. {remaining_nans} NaNs remain (likely groups with insufficient data).")
        # Optionally, fill any remaining NaNs (e.g., groups with only 1 point or all NaNs) with 0 or a global mean/median if desired
        # df[duty_col] = df[duty_col].fillna(0) # Example: Fill remaining with 0
    else:
        print(f"\nNo missing values found in '{duty_col}'.")

    if df_original_for_verify is not None:
        verify_interpolation(df_original_for_verify, df, duty_col, ['Reporting Economy', 'Product/Sector'])

    # Apply target transformation and calculate deltas
    group_cols = ['Reporting Economy', 'Product/Sector']
    target = None
    plot_xlabel = ''
    # Sets target var name + plot xlabel
    if target_transform == 'none':
        #print("\nUsing absolute change (Delta_ImportValue) as target.")
        target_value_col = 'ImportValue'
        df['Delta_TargetValue'] = df.groupby(group_cols)[target_value_col].diff()
        target = 'Delta_TargetValue'
        plot_xlabel = 'Change in Import Value (Absolute)'
    elif target_transform == 'log_diff':
        #print("\nApplying log transform and calculating log-difference as target.")
        non_positive_count = (df['ImportValue'] <= 0).sum()
        if non_positive_count > 0:
            print(f"Warning: Found {non_positive_count} non-positive ImportValue entries. Adding 1 before log transform.")
        df['Log_ImportValue'] = np.log(df['ImportValue'] + 1)
        target_value_col = 'Log_ImportValue'
        df['LogDiff_ImportValue'] = df.groupby(group_cols)[target_value_col].diff()
        target = 'LogDiff_ImportValue'
        plot_xlabel = 'Change in Log(Import Value + 1)'
    elif target_transform == 'pct_change':
        #print("\nCalculating percentage change as target.")
        df['PctChange_ImportValue'] = df.groupby(group_cols)['ImportValue'].pct_change()
        inf_count = np.isinf(df['PctChange_ImportValue']).sum()
        if inf_count > 0:
            print(f"Warning: Found {inf_count} infinite values in PctChange_ImportValue. Replacing with NaN.")
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
        target = 'PctChange_ImportValue'
        plot_xlabel = 'Percentage Change in Import Value'
    else:
        raise ValueError(f"Unknown target_transform: {target_transform}.")

    df['Delta_AverageDutyRate'] = df.groupby(group_cols)['AverageDutyRate'].diff()

    # Feature engineering (lags)
    numeric_features = ['Delta_AverageDutyRate']
    lag1_target_delta_col = None
    if use_lags:
        df['Lag1_Delta_AverageDutyRate'] = df.groupby(group_cols)['Delta_AverageDutyRate'].shift(1)
        lag1_target_delta_col = f'Lag1_{target}'
        df[lag1_target_delta_col] = df.groupby(group_cols)[target].shift(1)
        df['Lag1_AverageDutyRate'] = df.groupby(group_cols)['AverageDutyRate'].shift(1)
        numeric_features.extend(['Lag1_Delta_AverageDutyRate', lag1_target_delta_col, 'Lag1_AverageDutyRate'])
        #print("Using lagged features.")

    if outlier_percentile_threshold is not None and 0 < outlier_percentile_threshold < 0.5:
        # Calculate bounds on the target delta column *before* dropping NaNs from lags
        # but *after* calculating the delta itself. Handle potential NaNs in target here.
        target_vals_for_bounds = df[target].dropna()
        if not target_vals_for_bounds.empty:
            lower_bound = target_vals_for_bounds.quantile(outlier_percentile_threshold)
            upper_bound = target_vals_for_bounds.quantile(1 - outlier_percentile_threshold)
            print(f"\nApplying outlier removal based on percentiles: {outlier_percentile_threshold*100:.1f}% / {(1-outlier_percentile_threshold)*100:.1f}%")
            print(f"Removing '{target}' values outside range [{lower_bound:.4f}, {upper_bound:.4f}]")
            initial_rows = len(df)
            # Filter the dataframe
            df = df[(df[target] >= lower_bound) & (df[target] <= upper_bound) | df[target].isnull()].copy()
            # Keep rows where target is NaN for now, let the main dropna handle them based on features
            removed_rows = initial_rows - len(df)
            print(f"Identified {removed_rows} rows with target delta outside bounds.")
            print(f"Shape after outlier filtering (before final dropna): {df.shape}")
            if df.empty:
                print("Error: DataFrame empty after outlier removal attempt.")
                return None, None, None, None
        else:
            print("Warning: Could not calculate outlier bounds (target column might be all NaN). Skipping outlier removal.")
    elif outlier_percentile_threshold is not None:
         print("Warning: outlier_percentile_threshold should be between 0 and 0.5 (exclusive). Skipping outlier removal.")
    else:
        print("\nSkipping outlier removal.")

    # Data prep
    # Drop rows w/ NaNs created by diff/shift/transforms OR where lags are NaN
    cols_to_check_for_nan = [target] + numeric_features
    df_model = df.dropna(subset=cols_to_check_for_nan).copy()
    print(f"Shape after final dropna (removes NaNs from lags/diffs): {df_model.shape}")

    if df_model.empty:
        print("Error: No data remaining after calculating differences/lags and applying dropna.")
        return None, None, None, None

    # Plot target dist
    if plot_target_dist:
        plt.figure(figsize=(10, 6))
        data_to_plot = df_model[target]
        sns.histplot(data_to_plot, kde=True, bins=50)
        plt.title(f'Distribution of {target} (Outliers Removed: {outlier_percentile_threshold is not None})')
        plt.xlabel(plot_xlabel)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    categorical_features = ['Reporting Economy', 'Product/Sector']
    if not use_one_hot:
        categorical_features = []

    features = numeric_features + categorical_features
    X = df_model[features]
    y = df_model[target]

    # Preprocessing pipeline
    transformers = []
    if scale_numeric_features and numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
    
    if use_one_hot and categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features))
    
    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    else:
        preprocessor = 'passthrough'

    # Model pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            #min_samples_leaf=2,
            #min_samples_split=20,
            random_state=random_state,
            n_jobs=rf_n_jobs,
            oob_score=True
        ))
    ])

    # Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    #print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    #print(f"Training RandomForest model on target: {target}...")
    pipeline.fit(X_train, y_train)
    #print("Training complete.")
    try:
        oob = pipeline.named_steps['regressor'].oob_score_
        print(f"Model OOB Score: {oob:.4f}")
    except AttributeError: print("OOB score not available.")

    # Get feature names out
    try:
        feature_names_out = pipeline.named_steps['preprocessor'].get_feature_names_out()
    except Exception as e:
        print(f"Could not get transformed feature names automatically: {e}. Using original.")
        feature_names_out = features

    # Training predict
    metrics = {'train': {}, 'test': {}}
    y_pred_train = pipeline.predict(X_train)
    rmse_train = root_mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    metrics['train']['RMSE'] = rmse_train
    metrics['train']['R2_Score'] = r2_train
    #print(f"  Train Metrics: RMSE={rmse_train:.4f}, R2={r2_train:.4f}")

    # Test predict
    y_pred_test = pipeline.predict(X_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    metrics['test']['RMSE'] = rmse_test
    metrics['test']['R2_Score'] = r2_test
    #print(f"  Test Metrics:  RMSE={rmse_test:.4f}, R2={r2_test:.4f}")

    """if target_transform == 'log_diff':
        print(f"(Note: RMSE is on the scale of change in log(value+1))")
    elif target_transform == 'pct_change':
        print(f"(Note: RMSE is on the scale of percentage change, e.g., 0.1 = 10%)")
    else:
        print(f"(Note: RMSE is on the scale of absolute change in millions USD)")"""

    return pipeline, metrics, list(feature_names_out), features

def graph_feature_importance(pipeline):
    importances = pipeline.named_steps['regressor'].feature_importances_
    importance_series = pd.Series(importances, index=features_abs_or)
    top_n = 20
    plot_name = f'Top {top_n} Feature Importances'
    print('\n' + plot_name + ':')
    print(importance_series.sort_values(ascending=False).head(top_n))

    # Plot
    plt.figure(figsize=(10, max(5, top_n * 0.3)))
    importance_series.sort_values(ascending=False).head(top_n).plot(kind='barh')
    plt.title(plot_name)
    plt.xlabel('Importance')
    plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()

if __name__ == '__main__':
    FILE_PATH = 'data/merged_imports_duties.csv'

    # Absolute change (no outlier removal)
    print("\n--- Training: Absolute Change Target (NO Outlier Removal) ---")
    pipeline_abs, metrics_abs, _, orig_features_abs = train_import_delta_model(
        FILE_PATH,
        target_transform='none',
        outlier_percentile_threshold=None,
        plot_target_dist=False,
        rf_max_depth=10,
        use_lags=True
    )
    
    print(f"Result (No Outlier Removal):")
    print(f"  Train: R2={metrics_abs['train']['R2_Score']:.4f}, RMSE={metrics_abs['train']['RMSE']:.2f}")
    print(f"  Test:  R2={metrics_abs['test']['R2_Score']:.4f}, RMSE={metrics_abs['test']['RMSE']:.2f}")

    # Absolute change (outlier removal)
    print("\n--- Training: Absolute Change Target (WITH Outlier Removal) ---")
    pipeline_abs_or, metrics_abs_or, features_abs_or, orig_features_or = train_import_delta_model(
        FILE_PATH,
        target_transform='none',
        outlier_percentile_threshold=0.0005,
        plot_target_dist=False,
        rf_max_depth=10,
        use_lags=True
    )
    
    print(f"Result (WITH 0.1% Outlier Removal):")
    print(f"  Train: R2={metrics_abs_or['train']['R2_Score']:.4f}, RMSE={metrics_abs_or['train']['RMSE']:.2f}")
    print(f"  Test:  R2={metrics_abs_or['test']['R2_Score']:.4f}, RMSE={metrics_abs_or['test']['RMSE']:.2f}")

    #graph_feature_importance(pipeline_abs)
    #graph_feature_importance(pipeline_abs_or)

    MODEL_DIR = 'model'
    MODEL_PATH = os.path.join(MODEL_DIR, 'tariff_rf_pipeline.joblib')

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline_abs, MODEL_PATH)
    #joblib.dump(pipeline_abs_or, MODEL_PATH)
    print(f'\nSaved trained pipeline to: {MODEL_PATH}')
    
    # Assuming 'features' list holds the column names passed to the split
    input_features = orig_features_abs#orig_features_or#features_abs_or
    FEATURES_PATH = os.path.join(MODEL_DIR, 'model_features.joblib')
    joblib.dump(input_features, FEATURES_PATH)
    print(f'Saved input feature list to: {FEATURES_PATH}')
