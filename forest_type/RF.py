# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:27:16 2017

@author: acer
"""

import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.decomposition import PCA

PCA_COMPONENTS = 40

if __name__ == '__main__':
    print ("載入資料...")
    data = pd.read_csv("train.csv")
    print ("將資料分成CLASS 跟 ATTRIBUTE")
    train_label = []
    train_data = []
    print ("切割資料")
    train_label = data['Cover_Type']
    train_data = data.drop('Cover_Type', axis=1)
    train_data = train_data.drop('Id', axis=1)
    test_data = pd.read_csv("test.csv")
    
    print (test_data.head())
    test_label = test_data['Id']
    print (test_label)
    test_data =  test_data.drop('Id', axis=1)
    print (test_data.head())
    model= RandomForestClassifier(n_estimators=1000)
    # Train the model using the training sets and check score
    
    pca = PCA(n_components=25).fit(train_data)
    X_train_reduced = pca.transform(train_data)

    print ("生成隨機森林...")
    model.fit(X_train_reduced, train_label)
    print ("輸入資料建立模型....")
    #Predict Output
    print ("輸入test.csv 建立預測")
    
    X_test_reduced = pca.transform(test_data)
    predicted= model.predict(X_test_reduced)
    
    
    submission = pd.DataFrame({
        "Id": test_label,
        "Cover_Type": predicted
    })
    
    submission.to_csv("kaggle.csv", index=False)
