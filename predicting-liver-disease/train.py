from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

import pandas as pd
import numpy as np

from numpy import argmax

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

liver_df = pd.read_csv('./indian_liver_patient.csv')

liver_df = liver_df.dropna()

liver_df_normalised = liver_df.copy()

cleanup_nums = {"Gender":     {"Female": 1, "Male": 2}}

liver_df_normalised = liver_df_normalised.replace(cleanup_nums)

min_max_scaler = MinMaxScaler()

liver_df_normalised[liver_df_normalised.columns] = min_max_scaler.fit_transform(liver_df_normalised[liver_df_normalised.columns])

train, test = train_test_split(liver_df_normalised, test_size=0.2)

sigmoid = Sigmoid()

networkLayer = [10,20,10,1]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.3,0.6, 0.05,5000)

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
    feedForward.activate(row[:10])
    outputs = feedForward.getOutputs()
    print("Expected: {}, Actual: {}".format(int(row[10]),round(outputs[0])))
    if(int(row[10]) == int(round(outputs[0]))):
        totalCorrect = totalCorrect +1
    

percentageCorrect = totalCorrect/len(test.values) * 100

print(totalCorrect)

print("Percentage correct: {}%".format(percentageCorrect))

