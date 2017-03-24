# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:27:16 2017

@author: acer
"""

#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.decomposition import PCA

PCA_COMPONENTS = 100


def doWork(train, labels, test):
    print ("Converting training set to matrix")
    X_train = np.mat(train)

    print ("Fitting PCA. Components: %d" % PCA_COMPONENTS)
    pca = PCA(n_components=PCA_COMPONENTS).fit(X_train)

    print ("Reducing training to %d components" % PCA_COMPONENTS)
    X_train_reduced = pca.transform(X_train)

    print ("Fitting kNN with k=10, kd_tree")
    RF = RandomForestClassifier(n=1000)
    print (RF.fit(X_train_reduced, labels))

    print ("Reducing test to %d components" % PCA_COMPONENTS)
    X_test_reduced = pca.transform(test)

    print ("Preddicting numbers")
    predictions = RF.predict(X_test_reduced)

    print ("Writing to file")
    write_to_file(predictions)

    return predictions


def write_to_file(predictions):
    f = open("output-pca-knn-skilearn-v3.csv", "w")
    for p in predictions:
        f.write(str(p))
        f.write("\n")
    f.close()


if __name__ == '__main__':
    data = pd.read_csv("train.csv")
    print ("將資料分成CLASS 跟 ATTRIBUTE")
    labels = []
    train = []
    labels = data['label']
    train = data.drop('label', axis=1)
    
    test = pd.read_csv("test.csv")
    print (doWork(train, labels, test))

