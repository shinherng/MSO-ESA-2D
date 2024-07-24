import sys
import subprocess
import pkg_resources

required_packages = {
    'flask': 'Flask',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'scikit-learn': 'sklearn',
    'tensorflow': 'tensorflow',
    'matplotlib': 'matplotlib'
}

def install_missing_packages():
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    missing_packages = [required_packages[pkg] for pkg in required_packages if pkg not in installed_packages]
    
    if missing_packages:
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        print("All required packages are now installed.")
    else:
        print("All required packages are already installed.")

install_missing_packages()

from flask import Flask, render_template, request, redirect, url_for, session
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import io
import base64
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a real secret key in production

def convert_to_float(val):
    try:
        return float(val)
    except ValueError:
        return val


def safe_mape(y_true, y_pred, epsilon=1e-10):
    """
    Calculate Mean Absolute Percentage Error (MAPE) while handling zero and NaN values.
    
    Args:
    y_true (array-like): Array of true values
    y_pred (array-like): Array of predicted values
    epsilon (float): Small value to add to denominator to avoid division by zero
    
    Returns:
    float: MAPE value or None if calculation fails
    """
    try:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        logging.info(f"Shape of y_true: {y_true.shape}, Shape of y_pred: {y_pred.shape}")
        logging.info(f"y_true min: {np.nanmin(y_true)}, max: {np.nanmax(y_true)}")
        logging.info(f"y_pred min: {np.nanmin(y_pred)}, max: {np.nanmax(y_pred)}")
        
        # Remove NaN values
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        logging.info(f"Shape after removing NaNs - y_true: {y_true_clean.shape}, y_pred: {y_pred_clean.shape}")
        
        if len(y_true_clean) == 0 or len(y_pred_clean) == 0:
            logging.error("All values are NaN after cleaning")
            return None
        
        # Add epsilon to both arrays to handle zeros
        y_true_safe = y_true_clean + epsilon
        y_pred_safe = y_pred_clean + epsilon
        
        mape = mean_absolute_percentage_error(y_true_safe, y_pred_safe)
        logging.info(f"Calculated MAPE: {mape}")
        return mape
    except Exception as e:
        logging.error(f"Error in MAPE calculation: {str(e)}")
        return None


def calculate_mape(y_test, predictions, scaler, data):
    try:
        logging.info(f"Original y_test shape: {y_test.shape}")
        logging.info(f"Original predictions shape: {predictions.shape}")
        
        # Inverse transform y_test and predictions
        y_test_inv = scaler.inverse_transform(np.column_stack((y_test.reshape(-1, 1), np.zeros((len(y_test), len(data.columns) - 1)))))[:, 0]
        pred_inv = scaler.inverse_transform(np.column_stack((predictions, np.zeros((len(predictions), len(data.columns) - 1)))))[:, 0]
        
        logging.info(f"y_test after inverse_transform: {y_test_inv[:5]} ... {y_test_inv[-5:]}")
        logging.info(f"predictions after inverse_transform: {pred_inv[:5]} ... {pred_inv[-5:]}")
        
        mape = safe_mape(y_test_inv, pred_inv)
        if mape is not None:
            print(f"MAPE: {mape}")
        else:
            print("MAPE calculation failed. Check logs for details.")
        return mape
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"Exception in MAPE calculation: {str(e)}")
        return None


def export_predictions_to_csv(original_data, forecasts, target_variable, train_test_split, file_path='forecasts.csv'):
    """
    Export the original values, predictions, and forecasts to a CSV file.
    
    Args:
    original_data (pd.DataFrame): The original dataset
    predictions (np.array): The predicted values for the test set
    forecasts (np.array): The forecasted values for future time periods
    scaler (MinMaxScaler): The scaler used to normalize the data
    target_variable (str): The name of the target variable
    train_test_split (int): The index where the train-test split occurs
    file_path (str): The path where the CSV file will be saved
    
    Returns:
    None
    """
    # Prepare the original values
    original_values = original_data[target_variable].values.reshape(-1, 1)
    
    # Inverse transform the predictions and forecasts if they're not already
    # if predictions.ndim == 1:
    #     predictions = predictions.reshape(-1, 1)
    # if forecasts.ndim == 1:
    #     forecasts = forecasts.reshape(-1, 1)
    
    # if predictions.shape[1] == 1:
    #     predictions = scaler.inverse_transform(np.column_stack((predictions, np.zeros((len(predictions), scaler.scale_.shape[0] - 1)))))[:, 0]
    # if forecasts.shape[1] == 1:
    #     forecasts = scaler.inverse_transform(np.column_stack((forecasts, np.zeros((len(forecasts), scaler.scale_.shape[0] - 1)))))[:, 0]
    
    # Create a DataFrame with the original values
    df_export = pd.DataFrame({
        'Value': original_values.flatten(),
        'Type': ['Training' if i < train_test_split else 'Test' for i in range(len(original_values))]
    })
    
    # Add the predictions to the DataFrame
    # df_predictions = pd.DataFrame({
    #     'Value': predictions,
    #     'Type': ['Predicted' for _ in range(len(predictions))]
    # })
    
    # Add the forecasts to the DataFrame
    df_forecasts = pd.DataFrame({
        'Value': forecasts,
        'Type': ['Forecast' for _ in range(len(forecasts))]
    })
    
    df_export = pd.concat([df_export, df_forecasts], ignore_index=True)
    
    # Reorder columns
    df_export = df_export[['Type', 'Value']]
    
    # Export to CSV
    df_export.to_csv(file_path, index=False)



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file_path = request.form['file_path']
        if file_path:
            if os.path.isfile(file_path) and file_path.endswith(('.csv', '.xlsx')):
                session['file_path'] = file_path
                return redirect(url_for('select_variables'))
            elif os.path.isdir(file_path):
                files = [f for f in os.listdir(file_path) if f.endswith(('.csv', '.xlsx'))]
                if files:
                    session['folder_path'] = file_path
                    return redirect(url_for('select_file'))
        return render_template('index.html', error="Invalid file or folder path")
    return render_template('index.html')

@app.route('/select_file', methods=['GET', 'POST'])
def select_file():
    folder_path = session.get('folder_path')
    if not folder_path:
        return redirect(url_for('index'))
    
    files = [f for f in os.listdir(folder_path) if f.endswith(('.csv', '.xlsx'))]
    
    if request.method == 'POST':
        selected_file = request.form['selected_file']
        session['file_path'] = os.path.join(folder_path, selected_file)
        return redirect(url_for('select_variables'))
    
    return render_template('select_file.html', files=files)

@app.route('/select_variables', methods=['GET', 'POST'])
def select_variables():
    file_path = session.get('file_path')
    if not file_path:
        return redirect(url_for('index'))
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    
    df = df.applymap(convert_to_float)
    
    variables = df.columns.tolist()
    
    if request.method == 'POST':
        target_variable = request.form['target_variable']
        predictor_variables = request.form.getlist('predictor_variables')
        train_test_split = int(request.form['train_test_split'])
        session['target_variable'] = target_variable
        session['predictor_variables'] = predictor_variables
        session['train_test_split'] = train_test_split
        return redirect(url_for('forecast'))
    
    return render_template('select_variables.html', variables=variables, data_length=len(df))

@app.route('/forecast')
def forecast():
    file_path = session.get('file_path')
    target_variable = session.get('target_variable')
    # predictor_variables = session.get('predictor_variables')
    train_test_split = session.get('train_test_split')
    
    if not all([file_path, target_variable, train_test_split]):
        return redirect(url_for('index'))
    
    # Load and preprocess data
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    df = df.applymap(convert_to_float)
    
    data = df[[target_variable]]
    
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #scaled_data = scaler.fit_transform(data)
    
    train = data[:train_test_split]
    test = data[train_test_split:]
    
    prophet_mape, prophet_pred = prophet(train, test)
    lstm_mape, lstm_pred = lstm(train, test)

    if prophet_mape > lstm_mape:
        mape = lstm_mape
        forecasts = lstm_pred
    else:
        mape = prophet_mape
        forecasts = prophet_pred

    # Prepare training data
    # X_train, y_train = [], []
    # for i in range(len(train_data) - 1):
    #     X_train.append(train_data[i, :])
    #     y_train.append(train_data[i + 1, 0])
    # X_train, y_train = np.array(X_train), np.array(y_train)
    # X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    
    # Build and train the model
    # model = Sequential([
    #     LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    #     Dense(1)
    # ])
    # model.compile(optimizer='adam', loss='mse')
    # model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Make predictions on test data
    # mape = None
    # predictions = []
    # if len(test_data) > 0:
    #     last_sequence = train_data[-1:].reshape(1, 1, -1)
    #     for _ in range(len(test_data)):
    #         next_pred = model.predict(last_sequence)
    #         predictions.append(next_pred[0, 0])
    #         # Update the last sequence for the next prediction
    #         last_sequence = np.roll(last_sequence, -1, axis=2)
    #         last_sequence[0, 0, -1] = next_pred[0, 0]
        
    #     predictions = np.array(predictions)
    #     y_test = test_data[:, 0]  # Actual test values
        
    #     mape = calculate_mape(y_test, predictions, scaler, data)

    # # Make single-step forecasts for the next 10 time periods
    # last_sequence = scaled_data[-1:].reshape(1, 1, -1)
    # forecasts = []
    # for _ in range(10):
    #     next_pred = model.predict(last_sequence)
    #     forecasts.append(next_pred[0, 0])
    #     # Update the last sequence for the next prediction
    #     last_sequence = np.roll(last_sequence, -1, axis=2)
    #     last_sequence[0, 0, -1] = next_pred[0, 0]
    
    # forecasts = np.array(forecasts)

    # Export predictions and forecasts to CSV
    export_predictions_to_csv(df, forecasts, target_variable, train_test_split)

    # Inverse transform predictions and forecasts for plotting
    # if len(predictions) > 0:
    #     predictions = scaler.inverse_transform(np.column_stack((predictions, np.zeros((len(predictions), len(data.columns) - 1)))))[:, 0]
    # forecast_values = scaler.inverse_transform(np.column_stack((forecasts, np.zeros((len(forecasts), len(data.columns) - 1)))))[:, 0]

    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[:train_test_split], df[target_variable][:train_test_split], label='Training Data')
    if len(test) > 0:
        plt.plot(df.index[train_test_split:], df[target_variable][train_test_split:], label='Test Data')
        plt.plot(df.index[train_test_split:], forecasts, label='Forecasts', color='red')
    # plt.plot(range(len(df), len(df) + 10), forecasts, label='Forecast', color='red')
    plt.title(f'LSTM Forecast for {target_variable}')
    plt.xlabel('Time Step')
    plt.ylabel(target_variable)
    plt.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    
    return render_template('forecast.html', plot_url=plot_url, target=target_variable, mape=mape, forecast_values=forecasts)


if __name__ == '__main__':
    app.run(debug=True)