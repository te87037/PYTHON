import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn import svm
PCA_COMPONENTS = 100


def doWork(train, labels, test):
    print ("Converting training set to matrix")
    X_train = np.mat(train)

    print ("Fitting PCA. Components: %d" % PCA_COMPONENTS)
    pca = PCA(n_components=PCA_COMPONENTS).fit(X_train)
    
    print ("Reducing training to %d components" % PCA_COMPONENTS)
    X_train_reduced = pca.transform(X_train)

    print ("Fitting RandomForest")
    RF = RandomForestClassifier()
    print (RF.fit(X_train_reduced, labels))

    print ("Reducing test to %d components" % PCA_COMPONENTS)
    X_test_reduced = pca.transform(test)

    print ("Preddicting numbers")
    predictions = RF.predict(X_test_reduced)

    print ("Writing to file")
    write_to_file(predictions)

    return predictions


def write_to_file(predictions):
    id = []
    count = 0
    for i in predictions:
        count = count+1;
        id.append(count)
    submission = pd.DataFrame({
        "ImageId": id,
        "Label": predictions
    })
    submission.to_csv("kaggle.csv", index=False)


if __name__ == '__main__':
    data = pd.read_csv("train.csv")
    print ("將資料分成CLASS 跟 ATTRIBUTE")
    labels = []
    train = []
    labels = data['label']
    train = data.drop('label', axis=1)
    
    test = pd.read_csv("test.csv")
    print (doWork(train, labels, test))
# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs