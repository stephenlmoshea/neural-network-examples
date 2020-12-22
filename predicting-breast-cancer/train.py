from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

from numpy import argmax

from sklearn.model_selection import train_test_split

from datetime import datetime

data = pd.DataFrame()
data = pd.read_csv("./data.csv")

data.dropna(axis=1, inplace=True)

data.drop(labels='id',axis=1,inplace=True)

values = data.drop(labels='diagnosis',axis=1)
targets = data.diagnosis

encoder = LabelEncoder()
encoder.fit(targets)
encoded_targets = encoder.fit_transform(targets)

column_maxes = values.max()
df_max = column_maxes.max()
column_mins = values.min()
df_min = column_mins.min()
normalized_df = (values - df_min) / (df_max - df_min)

# print(encoded_targets)

normalized_df["encoded_targets"] = encoded_targets

train, test = train_test_split(normalized_df, test_size=0.2)

sigmoid = Sigmoid()

networkLayer = [30,60,30,1]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.7,0.8, 0.05, 2000)

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
    feedForward.activate(row[:30])
    outputs = feedForward.getOutputs()
    print("Expected: {}, Actual: {}".format(int(row[30]),round(outputs[0])))
    if(int(row[30]) == int(round(outputs[0]))):
        totalCorrect = totalCorrect +1
    

percentageCorrect = totalCorrect/len(test.values) * 100

print(totalCorrect)

print("Percentage correct: {}%".format(percentageCorrect))




