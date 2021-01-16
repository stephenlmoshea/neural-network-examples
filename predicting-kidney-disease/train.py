from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

import pandas as pd
import numpy as np

from numpy import argmax

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

from sklearn.impute import SimpleImputer

from numpy import nan
from numpy import isnan

ckd_df = pd.read_csv('./kidney_disease.csv')

col_dict={"bp":"blood_pressure",
          "sg":"specific_gravity",
          "al":"albumin",
          "su":"sugar",
          "rbc":"red_blood_cells",
          "pc":"pus_cell",
          "pcc":"pus_cell_clumps",
          "ba":"bacteria",
          "bgr":"blood_glucose_random",
          "bu":"blood_urea",
          "sc":"serum_creatinine",
          "sod":"sodium",
          "pot":"potassium",
          "hemo":"hemoglobin",
          "pcv":"packed_cell_volume",
          "wc":"white_blood_cell_count",
          "rc":"red_blood_cell_count",
          "htn":"hypertension",
          "dm":"diabetes_mellitus",
          "cad":"coronary_artery_disease",
          "appet":"appetite",
          "pe":"pedal_edema",
          "ane":"anemia"}

ckd_df.rename(columns=col_dict, inplace=True)

ckd_df['diabetes_mellitus'] =ckd_df['diabetes_mellitus'].replace(to_replace={'\tno':'no','\tyes':'yes',' yes':'yes'})
ckd_df['coronary_artery_disease'] = ckd_df['coronary_artery_disease'].replace(to_replace='\tno',value='no')
ckd_df['white_blood_cell_count'] = ckd_df['white_blood_cell_count'].replace(to_replace='\t8400',value='8400')

ckd_df["classification"]=ckd_df["classification"].replace("ckd\t", "ckd")

ckd_df["white_blood_cell_count"]=ckd_df["white_blood_cell_count"].replace("\t?", np.nan)
ckd_df["red_blood_cell_count"]=ckd_df["red_blood_cell_count"].replace("\t?", np.nan)
ckd_df['diabetes_mellitus'] = ckd_df['diabetes_mellitus'].replace(to_replace={'\tno':'no','\tyes':'yes',' yes':'yes'})
ckd_df['coronary_artery_disease'] = ckd_df['coronary_artery_disease'].replace(to_replace='\tno',value='no')
ckd_df['white_blood_cell_count'] = ckd_df['white_blood_cell_count'].replace(to_replace='\t8400',value='8400')
ckd_df["packed_cell_volume"]= ckd_df["packed_cell_volume"].replace("\t?", np.nan)

for string_column in ["red_blood_cells","pus_cell","pus_cell_clumps","bacteria","hypertension","diabetes_mellitus","coronary_artery_disease","pedal_edema","anemia","appetite"]:
  ckd_df[string_column]=ckd_df[string_column].astype(str)

ckd_df['red_blood_cells']=ckd_df['red_blood_cells'].replace({'normal':1,'abnormal':0})
ckd_df['pus_cell']=ckd_df['pus_cell'].replace({'normal':1,'abnormal':0})
ckd_df['pus_cell_clumps']=ckd_df['pus_cell_clumps'].replace({'notpresent':0,'present':1})
ckd_df['bacteria']=ckd_df['bacteria'].replace({'notpresent':0,'present':1})
ckd_df['hypertension']=ckd_df['hypertension'].replace({'no':0,'yes':1})
ckd_df['diabetes_mellitus']=ckd_df['diabetes_mellitus'].replace({'no':0,'yes':1})
ckd_df['coronary_artery_disease']=ckd_df['coronary_artery_disease'].replace({'no':0,'yes':1})
ckd_df['pedal_edema']=ckd_df['pedal_edema'].replace({'no':0,'yes':1})
ckd_df['anemia']=ckd_df['anemia'].replace({'no':0,'yes':1})
ckd_df['appetite']=ckd_df['appetite'].replace({'poor':0,'good':1})
ckd_df['classification']=ckd_df['classification'].replace({'ckd':1,'notckd':0})

ckd_df.drop('id',axis=1,inplace=True)

values = ckd_df.values

imputer = SimpleImputer(missing_values=nan, strategy='mean')
# transform the dataset
transformed_values = imputer.fit_transform(values)

min_max_scaler = MinMaxScaler()

transformed_values = min_max_scaler.fit_transform(transformed_values)

train, test = train_test_split(transformed_values, test_size=0.2, stratify=transformed_values[:,24])

sigmoid = Sigmoid()

networkLayer = [24,48,24,1]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.3,0.6, 0.05)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

backpropagation.initialise()
result = backpropagation.train(train)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)

print("Training time {} minutes".format(backpropagation.getTrainingTime()/60))

feedForward.save('./network.txt')

feedForward = FeedForward.load('./network.txt')

totalCorrect = 0
for num,row in enumerate(test):  
    feedForward.activate(row[0:24])
    outputs = feedForward.getOutputs()
    print("Expected: {}, Actual: {}".format(int(row[24]),round(outputs[0])))
    if(int(row[24]) == int(round(outputs[0]))):
        totalCorrect = totalCorrect +1
    

percentageCorrect = totalCorrect/len(test) * 100

print(totalCorrect)

print("Percentage correct: {}%".format(percentageCorrect))

