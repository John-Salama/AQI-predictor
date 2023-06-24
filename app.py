from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import csv
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the Keras model
model = tf.keras.models.load_model('FinalISA (2).h5')

# Load the mean and standard deviation for AQI and PM2.5
pm25_mean= 53.906898011002966
pm25_std= 34.07538895084159
aqi_mean= 126.82352941176471
aqi_std= 38.15992365823657

@app.route('/', methods=['POST'])
# Define the function to get the next 12 predictions
def get_next_12_predictions():
    request_json = request.get_json()
    JsonToCSV(request_json)
    # Load the data from the CSV file
    for i in range(12):
        test_data = pd.read_csv('data.csv')
        output = predict()
        output = output.json
        #remove the first row from the csv file
        test_data.drop(test_data.index[0] , axis=0 , inplace=True)
        # Add one hour to the last datetime string
        last_datetime_str = test_data.iloc[-1]['Date'] + ',' + test_data.iloc[-1]['Time']
        new_date , new_Time = addHour(last_datetime_str)
        # Create a new row for the next hour
        test_data.loc[26] = {'Date': new_date , 'Time' : new_Time, 'PM2.5': round(output['Predicted PM2.5']), 'AQI':round(output['Predicted AQI'])}
        # Save the new data back to the CSV file
        test_data.to_csv('data.csv', index=False)
    response = test_data.iloc[-12:]
    return jsonify({'data' : response.to_dict('records')})


def predict():
    # Get the file path from the request
    file = 'data.csv'
    # Apply the deployment script to the file
    test_data = pd.read_csv(file)      
    test_data['datetime'] = test_data.Date.map(str) + " " + test_data.Time
    test_data.datetime = pd.to_datetime(test_data.datetime)
    test_data.sort_values(by='datetime', inplace=True)
    test_data = test_data.drop(['Time', 'Date'], axis=1)
    test_data = test_data.set_index('datetime')
    new_filename = 'modified_data.csv'
    test_data.to_csv(new_filename)
    data = pd.read_csv('modified_data.csv', index_col='datetime', parse_dates=True)
    input_data2 = data[['PM2.5', 'AQI']]
    input_data2['AQI'] = (input_data2['AQI'] - aqi_mean) / aqi_std
    input_data2['PM2.5'] = (input_data2['PM2.5'] - pm25_mean) / pm25_std
    input_data = input_data2.values
    data = input_data.reshape(1, 25, 2)
    predictions = model.predict(data)
    print(predictions)
    aqi_pred = predictions[0][-1][0]
    pm25_pred = predictions[0][-1][1]
    aqi_pred_denorm = aqi_pred * aqi_std + aqi_mean
    pm25_pred_denorm = pm25_pred * pm25_std + pm25_mean

    # Return the predictions as a JSON response
    return jsonify({'Predicted AQI': aqi_pred_denorm, 'Predicted PM2.5': pm25_pred_denorm})


def JsonToCSV(request_json):
    # Load the JSON data
    data = request_json
        
    # Open a CSV file for writing
    with open('data.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(data[0].keys())


        # Write the data rows
        for row in data:
            writer.writerow(row.values())


def addHour(datetime_str):
    # Convert the datetime string to a datetime object
    datetime_obj = datetime.strptime(datetime_str, '%m/%d/%Y,%I:%M:%S %p')

    # Add one hour to the datetime object
    new_datetime_obj = datetime_obj + timedelta(hours=1)

    # Convert the datetime object back to a string
    new_datetime_str = new_datetime_obj.strftime('%m/%d/%Y,%I:%M:%S %p')

    # Print the new datetime string
    return new_datetime_str.split(',')

if __name__ == '__main__':
    app.run()