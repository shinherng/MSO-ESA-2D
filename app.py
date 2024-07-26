from flask import Flask, render_template, request, redirect, url_for, session, send_file
import pandas as pd
import numpy as np
from ML_model import aaron_lstm, aaron_rnn, ys_prophet
import plotly.graph_objs as go
import plotly.utils
import json
import uuid
import os
import logging

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# Set up logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

# Global dictionary to store results
results_store = {}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file_path = request.form['file_path']
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                return render_template('upload.html', error="Invalid file format. Please use CSV or Excel.")
            
            if len(df) < 50:
                return render_template('upload.html', error="File should contain at least 50 numerical values.")
            
            session['file_path'] = file_path
            return redirect(url_for('select_column'))
        except Exception as e:
            return render_template('upload.html', error=f"Error reading file: {str(e)}")
    return render_template('upload.html')

@app.route('/select_column', methods=['GET', 'POST'])
def select_column():
    file_path = session.get('file_path')
    if not file_path:
        return redirect(url_for('upload_file'))
    
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if request.method == 'POST':
        selected_column = request.form['column']
        train_test_split = int(request.form['train_test_split'])
        
        session['selected_column'] = selected_column
        session['train_test_split'] = train_test_split
        
        return redirect(url_for('forecast'))
    
    return render_template('select_column.html', columns=numeric_columns, total_rows=len(df))

@app.route('/forecast')
def forecast():
    file_path = session.get('file_path')
    selected_column = session.get('selected_column')
    train_test_split = session.get('train_test_split')
    
    if not all([file_path, selected_column, train_test_split]):
        return redirect(url_for('upload_file'))
    
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    data = df[selected_column].values
    
    train_data = pd.DataFrame({'Date': df.index[:train_test_split], 'Open': data[:train_test_split]})
    test_data = pd.DataFrame({'Date': df.index[train_test_split:], 'Open': data[train_test_split:]})
    
    lstm_pred, lstm_mape = aaron_lstm(train_data, test_data)
    rnn_pred, rnn_mape = aaron_rnn(train_data, test_data)
    prophet_pred, prophet_mape = ys_prophet(train_data['Open'], test_data['Open'], 10)
    
    best_model = min([(lstm_mape, 'LSTM'), (rnn_mape, 'RNN'), (prophet_mape, 'Prophet')], key=lambda x: x[0])
    best_mape, best_model_name = best_model
    
    if best_model_name == 'LSTM':
        predictions = lstm_pred
    elif best_model_name == 'RNN':
        predictions = rnn_pred
    else:
        predictions = prophet_pred
    
    # Generate a unique ID for this set of results
    results_id = str(uuid.uuid4())
    
    # Store the results in the global dictionary
    results_store[results_id] = {
        'original_data': data.tolist(),
        'predictions': predictions.flatten().tolist(),
        'best_model': best_model_name,
        'selected_column': selected_column,
        'train_test_split': train_test_split
    }
    
    # Prepare data for plotting
    trace_train = go.Scatter(x=df.index[:train_test_split], y=data[:train_test_split], mode='lines', name='Train')
    trace_test = go.Scatter(x=df.index[train_test_split:], y=data[train_test_split:], mode='lines', name='Test')
    trace_pred = go.Scatter(x=df.index[train_test_split:], y=predictions.flatten(), mode='lines', name='Predictions')
    
    layout = go.Layout(title='Time Series Forecast', xaxis=dict(title='Date'), yaxis=dict(title='Value'))
    fig = go.Figure(data=[trace_train, trace_test, trace_pred], layout=layout)
    
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('forecast.html', graph_json=graph_json, mape=best_mape, model=best_model_name, results_id=results_id)

@app.route('/download_results/<results_id>')
def download_results(results_id):
    try:
        if results_id not in results_store:
            logging.error(f"Results ID {results_id} not found in results_store")
            return redirect(url_for('upload_file'))
        
        results = results_store[results_id]
        
        original_data = results['original_data']
        predictions = results['predictions']
        train_test_split = results['train_test_split']
        
        logging.info(f"Original data length: {len(original_data)}, Predictions length: {len(predictions)}")
        
        # Create a DataFrame with actual data
        df = pd.DataFrame({
            'Actual': original_data,
            'Forecast': [np.nan] * len(original_data)  # Initialize forecast column with NaN
        })
        
        # Add predictions to the forecast column starting from the train_test_split index
        df.loc[train_test_split:, 'Forecast'] = predictions[:len(df) - train_test_split]
        
        filename = f"{results['best_model']}_forecasts_{results['selected_column']}.csv"
        file_path = os.path.join(os.path.dirname(__file__), filename)
        
        df.to_csv(file_path, index=False)
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )
    except Exception as e:
        logging.error(f"Error in download_results: {str(e)}")
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)