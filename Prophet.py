import sys
import subprocess
import pkg_resources
from flask import Flask, render_template, request, redirect, url_for, session
import os
import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
import io
import base64
import logging

logging.basicConfig(level=logging.INFO)

required_packages = {
    'flask': 'Flask',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'fbprophet': 'prophet',
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

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a real secret key in production

def convert_to_float(val):
    try:
        return float(val)
    except ValueError:
        return val

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
        session['target_variable'] = target_variable
        return redirect(url_for('forecast'))
    
    return render_template('select_variables.html', variables=variables, data_length=len(df))

@app.route('/forecast')
def forecast():
    file_path = session.get('file_path')
    target_variable = session.get('target_variable')
    
    if not all([file_path, target_variable]):
        return redirect(url_for('index'))
    
    # Load and preprocess data
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    df = df.applymap(convert_to_float)
    
    df.columns = df.columns.str.strip()
    df = df.iloc[::-1]
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    data = df.sort_values(by='Date')
    data.set_index('Date', inplace=True)
    monthly_data = data[target_variable].resample('M').last()
    monthly_returns = monthly_data.pct_change().dropna()
    df = monthly_returns.reset_index()
    df.columns = ['ds', 'y']
    
    train = df[df['ds'] <= '2016-12-31']
    actual_2017 = df[(df['ds'] >= '2017-01-01') & (df['ds'] <= '2017-12-31')]
    
    model = Prophet()
    model.fit(train)
    
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)
    forecast_2017 = forecast[(forecast['ds'] >= '2017-01-01') & (forecast['ds'] <= '2017-12-31')]
    
    results = pd.merge(actual_2017, forecast_2017[['ds', 'yhat']], on='ds')
    mape = np.mean(np.abs((results['y'] - results['yhat']) / results['y'])) * 100
    
    # Single-step forecasts for the next 10 time periods
    future_dates = model.make_future_dataframe(periods=10, freq='M', include_history=False)
    future_forecast = model.predict(future_dates)
    future_forecast_values = future_forecast[['ds', 'yhat']]

    plt.figure(figsize=(12, 6))
    plt.plot(df['ds'], df['y'], label='Actual')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red')
    plt.axvline(x=pd.Timestamp('2017-01-01'), color='black', linestyle='--', label='Forecast Start')
    plt.title('VOO Monthly Returns and Forecast')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    
    return render_template('forecast.html', plot_url=plot_url, mape=mape)

if __name__ == '__main__':
    app.run(debug=True)
