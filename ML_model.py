import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, LSTM

warnings.filterwarnings("ignore")


# def load_and_prepare_data(file_path, split_ratio=0.7):
#     data = pd.read_csv(file_path)
#     data = data.iloc[::-1]
#     length_data = len(data)
#     length_train = round(length_data * split_ratio)
#     length_test = length_data - length_train

#     train_data = data[:length_train].iloc[:, :2]
#     train_data['Date'] = pd.to_datetime(train_data['Date'])

#     test_data = data[length_train:].iloc[:,:2]
#     test_data['Date'] = pd.to_datetime(test_data['Date'])
    
#     return train_data, test_data, length_train, length_test

def scale_data(train_data, test_data):
    train_data.columns = train_data.columns.str.strip()
    dataset_train = train_data['Open'].values
    dataset_train = np.reshape(dataset_train, (-1, 1))

    total_datasets = len(dataset_train)
    time_step = total_datasets // 2

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
    scaled_dataset_validation = scaler.fit_transform(dataset_validation)

    x_test, y_test = [], []
    for i in range(time_step, len(scaled_dataset_validation)):
        x_test.append(scaled_dataset_validation[i-time_step:i, 0])
        y_test.append(scaled_dataset_validation[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
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

# def plot_training_history(history, title):
#     plt.figure(figsize=(10, 5))
#     plt.plot(history.history["loss"], label="Loss")
#     plt.plot(history.history["accuracy"], label="Accuracy")
#     plt.xlabel("Epochs")
#     plt.ylabel("Values")
#     plt.title(f"{title} - Loss and Accuracy vs Epoch")
#     plt.legend()
#     plt.show()

def calculate_metrics(y_true, y_pred):
    # rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # mse = mean_squared_error(y_true, y_pred)
    # mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

# def plot_predictions(y_true, y_pred, title):
#     plt.figure(figsize=(30, 10))
#     plt.plot(y_pred, label="Predicted", c="orange")
#     plt.plot(y_true, label="True", c="g")
#     plt.xlabel("Days")
#     plt.ylabel("Open Price")
#     plt.title(title)
#     plt.legend()
#     plt.show()

def aaron_lstm(train_data, test_data):
    # train_data, test_data, length_train, length_test = load_and_prepare_data("C:\\Users\\Aaron\\Downloads\\HistoricalPrices.csv")

    x_train, y_train, x_test, y_test, scaler = scale_data(train_data, test_data)
    lstm_model = build_lstm_model((x_train.shape[1], 1))
    y_test_scaled = scaler.inverse_transform(y_test)


    lstm_history = lstm_model.fit(x_train, y_train, epochs=10, batch_size=10)
    plot_training_history(lstm_history, "LSTM Model")


    y_pred_lstm = lstm_model.predict(x_test)
    y_pred_lstm = scaler.inverse_transform(y_pred_lstm)
    #plot_predictions(y_test_scaled, y_pred_lstm, "LSTM Model Predictions")

    lstm_metrics = calculate_metrics(y_test_scaled, y_pred_lstm)

    #print("LSTM Model Metrics (MAPE):", lstm_metrics)

    return y_pred_lstm, lstm_metrics

def aaron_rnn(train_data, test_data):
    x_train, y_train, x_test, y_test, scaler = scale_data(train_data, test_data)
    rnn_model = build_rnn_model((x_train.shape[1], 1))

    rnn_model.fit(x_train, y_train, epochs=50, batch_size=32)
    #plot_training_history(rnn_history, "Simple RNN Model")

    y_pred_rnn = rnn_model.predict(x_test)
    y_pred_rnn = scaler.inverse_transform(y_pred_rnn)
    y_test_scaled = scaler.inverse_transform(y_test)
    #plot_predictions(y_test_scaled, y_pred_rnn, "Simple RNN Model Predictions")

    rnn_metrics = calculate_metrics(y_test_scaled, y_pred_rnn)
    #print("RNN Model Metrics (MAPE):", rnn_metrics)

    return y_pred_rnn, rnn_metrics


def ys_prophet(train_data, test_data, periods):
    # Prepare data for Prophet
    train_data_df = pd.DataFrame({
        'ds': train_data.index,
        'y': train_data.values
    })

    model = Prophet()
    model.fit(train_data_df)
    periods = len(test_data)
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)
    train_test_split = len(train_data)
    predictions = forecast['yhat'].iloc[train_test_split:train_test_split + len(test_data)].values

    #plot_predictions(test_data.values, predictions, "Prophet Model Predictions")
    prophet_mape = calculate_metrics(test_data.values, predictions)
    #print("Prophet Model Metrics (MAPE):", prophet_mape)

    return predictions, prophet_mape

#def ys_prophet():
    #model = Prophet()
    #model.fit(train_data)
    #future = model.make_future_dataframe(periods=periods)
    #forecast = model.predict(future)
    #predictions = forecast['yhat'].iloc[train_test_split:train_test_split + len(test_data)].values
    #forecast_values = forecast['yhat'].iloc[train_test_split + len(test_data):].values
    #mape = np.mean(np.abs((results['y'] - results['yhat']) / results['y'])) * 100
    #print("Prophet Model Metrics (MAPE):", mape)
