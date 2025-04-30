import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify, Response, redirect, url_for
import plotly.graph_objects as go
import plotly.express as px

# --- Configuration ---
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "tariff_rf_pipeline.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.joblib")
DATA_PATH = "data/merged_imports_duties.csv"

feature_labels = {
    'Delta_AverageDutyRate': 'Change in Average Duty Rate This Year (%)',
    'Lag1_Delta_AverageDutyRate': 'Change in Average Duty Rate Last Year (%)',
    'Lag1_Delta_TargetValue': 'Change in Import Value Last Year (Millions USD)',  # Assumes model preds absolute change ('none' transform)
    'Lag1_AverageDutyRate': 'Average Duty Rate Last Year (%)'
}

# --- Load Model and Features ---
try:
    pipeline = joblib.load(MODEL_PATH)
    print("Model pipeline loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    pipeline = None
except Exception as e:
    print(f"Error loading model: {e}")
    pipeline = None

try:
    model_features = joblib.load(FEATURES_PATH)
    #print(f"Model expects features: {model_features}")
    # Separate numeric and categorical based on common prefixes/names
    numeric_features_expected = [f for f in model_features if f.startswith(('Delta_', 'Lag1_'))]
    categorical_features_expected = [f for f in model_features if f in ['Reporting Economy', 'Product/Sector']]
except FileNotFoundError:
    print(f"Warning: Features file not found at {FEATURES_PATH}. Input validation may be limited.")
    model_features = None
    numeric_features_expected = ['Delta_AverageDutyRate', 'Lag1_Delta_AverageDutyRate', 'Lag1_Delta_TargetValue', 'Lag1_AverageDutyRate'] # Fallback guess
    categorical_features_expected = ['Reporting Economy', 'Product/Sector'] # Fallback guess
except Exception as e:
    print(f"Error loading features: {e}")
    model_features = None
    numeric_features_expected = []
    categorical_features_expected = []

# --- Load Data for Dropdowns (Optional) ---
try:
    df_full = pd.read_csv(DATA_PATH)
    # Get unique sorted lists for dropdowns
    reporting_economies = sorted(df_full['Reporting Economy'].unique().tolist())
    product_sectors = sorted(df_full['Product/Sector'].unique().tolist())
    # Remove excluded economies if needed from dropdowns
    economies_to_exclude = ['World', 'United States of America', 'China'] # Match training
    reporting_economies = [e for e in reporting_economies if e not in economies_to_exclude]
    print("Loaded data for dropdowns.")
except FileNotFoundError:
    print(f"Warning: Data file not found at {DATA_PATH}. Dropdowns may be empty.")
    reporting_economies = ["Example Economy"] # Placeholder
    product_sectors = ["Example Sector"] # Placeholder
    df_full = None # Indicate data isn't loaded for plots
except Exception as e:
    print(f"Error loading data for dropdowns: {e}")
    reporting_economies = []
    product_sectors = []
    df_full = None

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Routes ---
@app.route('/', methods=['GET'])
def home():
    """Renders the main page with the input form."""
    # Pass lists for dropdowns to the template
    return render_template(
        'index.html',
        reporting_economies=reporting_economies,
        product_sectors=product_sectors,
        numeric_features=numeric_features_expected,
        feature_labels=feature_labels,
        prediction_result=None,
        form_values={}
    )

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # If someone tries to access /predict directly via GET,
        # simply redirect them back to the home page.
        return redirect(url_for('home'))

    """Handles form submission, makes prediction, and re-renders the page."""
    if pipeline is None:
        return render_template(
            'index.html',
            reporting_economies=reporting_economies,
            product_sectors=product_sectors,
            numeric_features=numeric_features_expected,
            feature_labels=feature_labels,
            prediction_result="Error: Model not loaded.",
            error="Model could not be loaded. Check server logs."
        )

    try:
        # 1. Extract data from form
        input_data = {}
        # Categorical
        input_data['Reporting Economy'] = request.form.get('reporting_economy')
        input_data['Product/Sector'] = request.form.get('product_sector')
        # Numeric - handle potential errors converting to float
        for feature in numeric_features_expected:
             try:
                 input_data[feature] = float(request.form.get(feature))
             except (TypeError, ValueError):
                 # Handle cases where input is missing or not a number
                 input_data[feature] = 0.0 # Or np.nan, but 0 might be safer default
                 print(f"Warning: Invalid or missing input for {feature}, using 0.0")


        # 2. Create DataFrame for prediction (must match training structure)
        # Ensure columns are in the same order as model_features if available
        predict_df = pd.DataFrame([input_data])
        if model_features:
             predict_df = predict_df[model_features] # Reorder/select columns
        else:
             # If features weren't loaded, hope the order is correct based on defaults
             print("Warning: Predicting based on default feature order.")


        # 3. Make prediction
        prediction = pipeline.predict(predict_df)
        predicted_delta = prediction[0] # Get the single prediction value

        # Format the result
        result_text = f"Predicted Change in Import Value: ${predicted_delta:,.2f} Million USD"
        error_text = None

    except Exception as e:
        print(f"Prediction Error: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback to console
        result_text = "Error during prediction."
        error_text = f"An error occurred: {e}"

    # Re-render the page with the result and original form values
    return render_template(
        'index.html',
        reporting_economies=reporting_economies,
        product_sectors=product_sectors,
        numeric_features=numeric_features_expected,
        feature_labels=feature_labels,
        prediction_result=result_text,
        error=error_text,
        # Pass back the user's selections to repopulate the form
        selected_economy=request.form.get('reporting_economy'),
        selected_sector=request.form.get('product_sector'),
        form_values=request.form # Pass all form values back
    )

@app.route('/plot_history', methods=['GET'])
def plot_history():
    print("\n--- Received request for /plot_history ---")
    if df_full is None:
        print("Error: df_full is None")
        return jsonify({"error": "Historical data not loaded."}), 404

    economy = request.args.get('economy')
    sector = request.args.get('sector')
    print(f"Economy: {economy}, Sector: {sector}")

    if not economy or not sector:
        print("Error: Missing economy or sector parameter")
        return jsonify({"error": "Economy and Sector parameters are required."}), 400

    # Filter data
    print("Filtering data...")
    plot_df = df_full[
        (df_full['Reporting Economy'] == economy) &
        (df_full['Product/Sector'] == sector)
    ].sort_values('Year')
    print(f"Filtered plot_df shape: {plot_df.shape}")

    if plot_df.empty:
         print("Error: No data found after filtering")
         return jsonify({"error": f"No data found for {economy} / {sector}."}), 404

    # Create Plotly figure
    print("Creating Plotly figure...")
    fig = go.Figure()
    # Plot Import Value - Convert data to list explicitly just in case
    fig.add_trace(go.Scatter(
        x=plot_df['Year'].tolist(), y=plot_df['ImportValue'].tolist(),
        mode='lines+markers', name='Import Value (Millions USD)', yaxis='y1'
    ))
    # Plot Average Duty Rate on secondary axis - Convert data to list
    fig.add_trace(go.Scatter(
        x=plot_df['Year'].tolist(), y=plot_df['AverageDutyRate'].tolist(),
        mode='lines+markers', name='Avg Duty Rate (%)', yaxis='y2'
    ))

    # Layout with secondary y-axis AND specified height
    fig.update_layout(
        template='plotly_dark',
        title=f'Historical Data for {economy} - {sector}',
        xaxis_title='Year',
        yaxis=dict(title='Import Value (Millions USD)'),
        yaxis2=dict(title='Avg Duty Rate (%)', overlaying='y', side='right'),
        legend=dict(x=0.1, y=1.1, orientation="h"),
        height=500 # Adjust height as needed
    )

    # --- Use fig.to_json() for serialization ---
    print("Converting figure to JSON using fig.to_json()...")
    graphJSON_string = fig.to_json()
    print("Sending JSON response.")
    # Return the JSON string with the correct content type
    return Response(graphJSON_string, mimetype='application/json')

@app.route('/get_latest_data', methods=['GET'])
def get_latest_data():
    """Fetches latest data needed to populate prediction inputs."""
    if df_full is None:
        return jsonify({"error": "Historical data not loaded."}), 500

    economy = request.args.get('economy')
    sector = request.args.get('sector')

    if not economy or not sector:
        return jsonify({"error": "Economy and Sector parameters are required."}), 400

    # Filter data for the specific group and sort by year descending
    group_df = df_full[
        (df_full['Reporting Economy'] == economy) &
        (df_full['Product/Sector'] == sector)
    ].sort_values('Year', ascending=False)

    if group_df.empty:
        return jsonify({"error": f"No historical data found for {economy} / {sector}."}), 404

    # Get the latest year (T)
    latest_row = group_df.iloc[0]
    latest_year = latest_row['Year']
    latest_import_value = latest_row['ImportValue']
    latest_duty_rate = latest_row['AverageDutyRate']

    # Initialize values needed for prediction inputs
    # For predicting T+1 assuming constant tariff from T
    delta_average_duty_rate_input = 0.0 # Constant tariff assumption
    lag1_delta_import_value_input = 0.0
    lag1_average_duty_rate_input = latest_duty_rate # Rate in year T
    lag1_delta_average_duty_rate_input = 0.0

    # Try to get the previous year (T-1) to calculate deltas for year T
    if len(group_df) > 1:
        prev_row = group_df.iloc[1]
        # Ensure the previous row is indeed the preceding year
        if prev_row['Year'] == latest_year - 1:
            prev_import_value = prev_row['ImportValue']
            prev_duty_rate = prev_row['AverageDutyRate']

            # This is the change that occurred *last year* (T-1 to T)
            lag1_delta_import_value_input = latest_import_value - prev_import_value
            # This is the change in duty rate that occurred *last year* (T-1 to T)
            lag1_delta_average_duty_rate_input = latest_duty_rate - prev_duty_rate
        else:
             print(f"Warning: Gap in years for {economy}/{sector}. Previous row year: {prev_row['Year']}, Latest: {latest_year}. Using 0 for deltas.")
    else:
        print(f"Warning: Only one year of data found for {economy}/{sector}. Using 0 for deltas.")


    # Prepare the data to send back
    # Keys should match the 'name' attributes of your input fields in HTML
    data_for_inputs = {
        'Delta_AverageDutyRate': delta_average_duty_rate_input,
        'Lag1_Delta_AverageDutyRate': lag1_delta_average_duty_rate_input,
        'Lag1_Delta_TargetValue': lag1_delta_import_value_input, # Assuming target is absolute change
        'Lag1_AverageDutyRate': lag1_average_duty_rate_input
    }

    return jsonify(data_for_inputs)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='127.0.0.1', port=port)
