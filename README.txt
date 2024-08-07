Introduction:
This python flask app provides a web user interface to generate forecasts for time series data using 3 methods, Long Short-Term Memory network (LSTM), Recurrent Neural Network (RNN) and the Prophet time series model. This zip contains python and html scripts. Python is required to run the app, also ensure that python is added as a path variable. Ensure the csv/excel file to be forecasted contains columns of numbers without NA values, where each column is a single variable. The column to be forecasted needs to have at least 50 values as this is required for the LSTM model to work.

How to run:
1) Open up the command prompt and change working directory to the folder where the files are stored locally
2) Run "python app.py" in the command prompt
3) Using a web browser, access this link http://127.0.0.1:5000/
4) Input the file path of a csv/excel file to be forecasted
5) Click on the "PROCEED" button
6) Choose the column of time series data to be forecasted from the dropdown box
7) Choose the train/test split ratio by sliding the circle indicator
8) Click on the "PROCEED" button
9) Either download the results or restart the program or exit it


Required packages in Python:
1) flask
2) pandas
3) numpy
4) plotly
5) json
6) uuid
7) os
8) logging
9) sklearn
10) prophet
11) keras