from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

from numpy import argmax

from sklearn.model_selection import train_test_split

from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

data = pd.DataFrame()
data = pd.read_csv('./heart.csv')

data.dropna(axis=1, inplace=True)

min_max_scaler = MinMaxScaler()

data[["age", "cp", "trestbps", "chol", "thalach", "oldpeak", "slope", "thal"]] = min_max_scaler.fit_transform(data[["age", "cp", "trestbps", "chol", "thalach", "oldpeak", "slope", "thal"]])

train, test = train_test_split(data, test_size=0.2)

sigmoid = Sigmoid()

networkLayer = [13,26,13,1]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.3,0.5, 0.01, 2000)

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

totalCorrect = 0
for num,row in enumerate(test.values):
    feedForward.activate(row[:13])
    outputs = feedForward.getOutputs()
    print("Expected: {}, Actual: {}".format(int(row[13]),round(outputs[0])))
    if(int(row[13]) == int(round(outputs[0]))):
        totalCorrect = totalCorrect +1
    

percentageCorrect = totalCorrect/len(test.values) * 100

print(totalCorrect)

print("Percentage correct: {}%".format(percentageCorrect))




