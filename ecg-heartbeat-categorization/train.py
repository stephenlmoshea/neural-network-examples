from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

import pandas as pd
import numpy as np

from numpy import argmax

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

mit_test_data = pd.read_csv('mitbih_test.csv', header=None)

label_names = {0 : 'N',
              1: 'S',
              2: 'V',
              3: 'F',
              4 : 'Q'}

inputs = mit_test_data.iloc[:, :187]

targets = mit_test_data.iloc[:, 187:]

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(targets.values.reshape(-1,))

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(targets), 1)
onehot_encoded = onehot_encoder.fit_transform(targets)

outputs_df = pd.DataFrame.from_records(onehot_encoded)

trainingSet = pd.concat([inputs, outputs_df], axis=1)

train, test = train_test_split(trainingSet, test_size=0.2)

sigmoid = Sigmoid()

networkLayer = [187,50,50,5]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.7,0.8, 0.05, 1)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

backpropagation.initialise()
result = backpropagation.train(train.to_numpy().tolist())

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)

print("Training time {} minutes".format(backpropagation.getTrainingTime()/60))

feedForward.save('./network.txt')

feedForward = FeedForward.load('./network.txt')

totalCorrect = 0
for num,row in enumerate(test.values):
    feedForward.activate(row[0:187])
    outputs = feedForward.getOutputs()
    actualClass = label_encoder.inverse_transform([argmax(outputs)])
    expectedClass = label_encoder.inverse_transform([argmax(row[187:])])

    print("Expected: {}, Predicted: {}".format(label_names[int(expectedClass)],label_names[int(actualClass)]))
    if(expectedClass == actualClass):
        totalCorrect = totalCorrect +1
    

percentageCorrect = totalCorrect/len(test.values) * 100

print(totalCorrect)

print("Percentage correct: {}%".format(percentageCorrect))
