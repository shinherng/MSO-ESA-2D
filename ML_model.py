import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, LSTM

def scale_data(train_data, test_data):
    train_data.columns = train_data.columns.str.strip()
    dataset_train = train_data['Open'].values
    dataset_train = np.reshape(dataset_train, (-1, 1))

    total_datasets = len(dataset_train)
    time_step = max(1, total_datasets // 2)  # Ensure time_step is at least 1

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_train_scaled = scaler.fit_transform(dataset_train)

    x_train, y_train = [], []
    for i in range(time_step, len(dataset_train_scaled)):
        x_train.append(dataset_train_scaled[i-time_step:i, 0])
        y_train.append(dataset_train_scaled[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], 1))

    test_data.columns = test_data.columns.str.strip()
    dataset_validation = test_data['Open'].values
    dataset_validation = np.reshape(dataset_validation, (-1, 1))
    scaled_dataset_validation = scaler.transform(dataset_validation)

    x_test, y_test = [], []
    for i in range(time_step, len(scaled_dataset_validation)):
        x_test.append(scaled_dataset_validation[i-time_step:i, 0])
        y_test.append(scaled_dataset_validation[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    
    # Check if x_test is not empty before reshaping
    if x_test.size > 0:
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    else:
        x_test = np.array([])  # Return an empty array if there's not enough test data
    
    y_test = np.reshape(y_test, (-1, 1))

    return x_train, y_train, x_test, y_test, scaler, time_step

def build_rnn_model(input_shape, units):
    regressor = Sequential()
    regressor.add(SimpleRNN(units=units, activation="tanh", return_sequences=True, input_shape=input_shape))
    regressor.add(Dropout(0.2))
    regressor.add(SimpleRNN(units=units, activation="tanh", return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(SimpleRNN(units=units, activation="tanh", return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(SimpleRNN(units=units))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer="adam", loss="mean_absolute_percentage_error", metrics=["accuracy"])
    return regressor

def build_lstm_model(input_shape):
    model_lstm = Sequential()
    model_lstm.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model_lstm.add(LSTM(64, return_sequences=False))
    model_lstm.add(Dense(32))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss="mean_absolute_percentage_error", optimizer="adam", metrics=["accuracy"])
    return model_lstm

def calculate_metrics(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

def aaron_lstm(train_data, test_data):
    x_train, y_train, x_test, y_test, scaler, time_step = scale_data(train_data, test_data)
    
    if x_test.size == 0:
        return np.array([]), np.inf  # Return empty array and infinite MAPE if there's not enough test data
    
    lstm_model = build_lstm_model((x_train.shape[1], 1))
    lstm_model.fit(x_train, y_train, epochs=10, batch_size=10, verbose=0)
    
    y_pred_lstm = lstm_model.predict(x_test)
    y_pred_lstm = scaler.inverse_transform(y_pred_lstm)
    y_test_scaled = scaler.inverse_transform(y_test)
    
    lstm_metrics = calculate_metrics(y_test_scaled, y_pred_lstm)
    
    return y_pred_lstm, lstm_metrics

def aaron_rnn(train_data, test_data):
    x_train, y_train, x_test, y_test, scaler, time_step = scale_data(train_data, test_data)
    
    if x_test.size == 0:
        return np.array([]), np.inf  # Return empty array and infinite MAPE if there's not enough test data
    
    rnn_model = build_rnn_model((x_train.shape[1], 1), 50)
    rnn_model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    y_pred_rnn = rnn_model.predict(x_test)
    y_pred_rnn = scaler.inverse_transform(y_pred_rnn)
    y_test_scaled = scaler.inverse_transform(y_test)
    
    rnn_metrics = calculate_metrics(y_test_scaled, y_pred_rnn)
    
    return y_pred_rnn, rnn_metrics

def ys_prophet(train_data, test_data, periods):
    # Prepare data for Prophet
    train_data_df = pd.DataFrame({
        'ds': pd.date_range(start='2000-01-01', periods=len(train_data), freq='D'),
        'y': train_data
    })

    model = Prophet()
    model.fit(train_data_df)
    
    future = model.make_future_dataframe(periods=len(test_data))
    forecast = model.predict(future)
    
    predictions = forecast['yhat'].iloc[-len(test_data):].values
    
    if len(test_data) > 0:
        prophet_mape = calculate_metrics(test_data, predictions)
    else:
        prophet_mape = np.inf  # Return infinite MAPE if there's no test data
    
    return predictions, prophet_mape