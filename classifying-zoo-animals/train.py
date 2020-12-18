from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

import pandas as pd
import numpy as np

from numpy import argmax

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

zoo = pd.read_csv("./zoo.csv")
classes = pd.read_csv("./class.csv")
class_lookup = classes.set_index('Class_Number').to_dict(orient='index')

y_data = zoo.iloc[:,-1:]

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_data['class_type'])

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

outputs_df = pd.DataFrame.from_records(onehot_encoded)

concatenated_dataframes = pd.concat([zoo, outputs_df], axis=1)

train, test = train_test_split(concatenated_dataframes, test_size=0.3, random_state=42, stratify=concatenated_dataframes.iloc[:,18:])

train_name = train['animal_name']
test_name = test['animal_name']

train_class = train['class_type']
test_class = test['class_type']

train_x = train.iloc[:,1:]
train_x = train_x.drop('class_type',1)
test_x = test.iloc[:,1:]
test_x = test_x.drop('class_type',1)

sigmoid = Sigmoid()

networkLayer = [16,32,24,7]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.7,0.8, 0.005, 2000)

backpropagation.initialise()
result = backpropagation.train(train_x.to_numpy().tolist())

feedForward.save('./network.txt')

feedForward = FeedForward.load('./network.txt')

totalCorrect = 0
for num,row in enumerate(test_x.values):
    feedForward.activate(row[0:16])
    outputs = feedForward.getOutputs()
    actualClass = label_encoder.inverse_transform([argmax(outputs)])
    expectedClass = label_encoder.inverse_transform([argmax(row[16:])])
    print("Animal: {}, Expected: {}, Predicted: {}".format(test_name.iloc[num],class_lookup[expectedClass[0]]['Class_Type'],class_lookup[actualClass[0]]['Class_Type']))
    if(expectedClass == actualClass):
        totalCorrect = totalCorrect +1
    

percentageCorrect = totalCorrect/len(test.values) * 100

print(totalCorrect)

print("Percentage correct: {}%".format(percentageCorrect))


