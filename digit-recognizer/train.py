from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

from numpy import argmax

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from datetime import datetime

data = pd.DataFrame()
data = pd.read_csv("./train.csv")

min_max_scaler = MinMaxScaler()

data[data.loc[:, data.columns != 'label'].columns] = min_max_scaler.fit_transform(data[data.loc[:, data.columns != 'label'].columns])

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data['label'])

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

outputs_df = pd.DataFrame.from_records(onehot_encoded)

concatenated_dataframes = pd.concat([data, outputs_df], axis=1)

train, test = train_test_split(concatenated_dataframes, test_size=0.3, random_state=42, stratify=concatenated_dataframes["label"])

train_class = train['label']
test_class = test['label']

train = train.drop('label',1)
test = test.drop('label',1)

print(train.shape)

sigmoid = Sigmoid()

networkLayer = [784,32,10]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.7,0.8, 0.05, 2)

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
    feedForward.activate(row[0:784])
    outputs = feedForward.getOutputs()
    actualClass = label_encoder.inverse_transform([argmax(outputs)])
    expectedClass = label_encoder.inverse_transform([argmax(row[784:])])

    print("Expected: {}, Predicted: {}".format(int(expectedClass),int(actualClass)))
    if(expectedClass == actualClass):
        totalCorrect = totalCorrect +1
    

percentageCorrect = totalCorrect/len(test.values) * 100

print(totalCorrect)

print("Percentage correct: {}%".format(percentageCorrect))


