from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

import pandas as pd
import numpy as np

from numpy import argmax

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

pd_speech_df = pd.read_csv('./pd_speech_features.csv')

pd_speech_df = pd_speech_df.dropna()

pd_speech_df_normalised = pd_speech_df.copy()

min_max_scaler = MinMaxScaler()

pd_speech_df_normalised[pd_speech_df_normalised.loc[:, pd_speech_df_normalised.columns != 'class'].columns] = min_max_scaler.fit_transform(pd_speech_df_normalised[pd_speech_df_normalised.loc[:, pd_speech_df_normalised.columns != 'class'].columns])

train, test = train_test_split(pd_speech_df_normalised, test_size=0.2, stratify=pd_speech_df_normalised['class'])

sigmoid = Sigmoid()

networkLayer = [754,70,40,1]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.3,0.6, 0.05,50)

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
    feedForward.activate(row[:754])
    outputs = feedForward.getOutputs()
    print("Expected: {}, Actual: {}".format(int(row[754]),round(outputs[0])))
    if(int(row[754]) == int(round(outputs[0]))):
        totalCorrect = totalCorrect +1
    

percentageCorrect = totalCorrect/len(test.values) * 100

print(totalCorrect)

print("Percentage correct: {}%".format(percentageCorrect))

