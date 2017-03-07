# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:27:16 2017

@author: acer
"""

#Import Library
from sklearn.ensemble import RandomForestClassifier 
import pandas as pd

#載入函式庫
print ("載入資料...")
data = pd.read_csv("train.csv")
print ("將資料分成CLASS 跟 ATTRIBUTE")
train_label = []
train_data = []
print ("切割資料")
train_label = data['label']
train_data = data.drop('label', axis=1)

model= RandomForestClassifier(n_estimators=1000)
# Train the model using the training sets and check score
print ("生成隨機森林...")
model.fit(train_data, train_label)
print ("輸入資料建立模型....")
#Predict Output
print ("輸入test.csv 建立預測")
test_data = pd.read_csv("test.csv")
predicted= model.predict(test_data)
id = []
count = 0
for i in predicted:
        count.append(i)
submission = pd.DataFrame({
        "ImageId": id,
        "Label": predicted
    })
    
submission.to_csv("kaggle.csv", index=False)