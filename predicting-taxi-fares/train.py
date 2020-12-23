from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

from utils import preprocess, feature_engineer

import pandas as pd
import numpy as np

from numpy import argmax

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('NYC_taxi.csv', parse_dates=['pickup_datetime'], nrows=500000)

# Perform preprocessing and feature engineering
df = preprocess(df)
df = feature_engineer(df)

# Scale the features
df_prescaled = df.copy()
df_scaled = df.copy()
df_scaled = df.drop(['fare_amount'], axis=1)

targets = df.fare_amount.values.reshape(-1, 1)

min_max_scaler = MinMaxScaler()

df_scaled[df_scaled.columns] = min_max_scaler.fit_transform(df_scaled[df_scaled.columns])

target_scaler = MinMaxScaler()
target_scaler.fit(targets)

targets_scaled = target_scaler.transform(targets)

df_scaled["targets_scaled"] = targets_scaled

train, test = train_test_split(df_scaled, test_size=0.2)

sigmoid = Sigmoid()

networkLayer = [17,34,17,1]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.7,0.8, 0.05, 1)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

backpropagation.initialise()
result = backpropagation.train(train.values)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)

print("Training time {} minutes".format(backpropagation.getTrainingTime()/60))

feedForward.save('./network.txt')

feedForward = FeedForward.load('./network.txt')

def predict_random(df_prescaled, X_test, model, target_scaler):
    sample = X_test.sample(n=1, random_state=np.random.randint(low=0, high=10000))
    idx = sample.index[0]

    actual_fare = df_prescaled.loc[idx,'fare_amount']
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_of_week = day_names[df_prescaled.loc[idx,'day_of_week']]
    hour = df_prescaled.loc[idx,'hour']

    model.activate(sample.values[0][0:17])
    outputs = feedForward.getOutputs()

    predicted_fare = float(outputs[0])
   
    predicted_fare_original = target_scaler.inverse_transform([[predicted_fare]])[0][0]
    rmse = np.sqrt(np.square(predicted_fare_original-actual_fare))

    print("Trip Details: {}, {}:00hrs".format(day_of_week, hour))  
    print("Actual fare: ${:0.2f}".format(actual_fare))
    print("Predicted fare: ${:0.2f}".format(predicted_fare_original))
    print("RMSE: ${:0.2f}".format(rmse))

predict_random(df_prescaled, test, feedForward, target_scaler)