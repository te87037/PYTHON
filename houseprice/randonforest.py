"""
Created on Tue Jan 17 09:27:16 2017

@author: acer
"""

#Import Library
from sklearn.ensemble import RandomForestClassifier 
import pandas as pd
from  sklearn.preprocessing import LabelEncoder
#載入函式庫
print ("載入資料...")
data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
print ("將資料分成CLASS 跟 ATTRIBUTE")
train_label = []
train_data = []
print ("切割資料")

categorical_fields=['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
                    'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                    'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual',
                    'GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition'
                    ]

numerical_fields=['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                  'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                  'TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
                  'MiscVal','MoSold','YrSold']

ally_dict={'NA':0,'Grvl':1,'Pave':2} ## dictionary to digitize the field : Alley

##空値資料，數值型用0取代，類別行用預設取代
def pre_process_data():
    print ("資料預處理...")
    for col in categorical_fields:
        data[col].fillna('default',inplace=True)
        test_data[col].fillna('default',inplace=True)

    for col in numerical_fields:
        data[col].fillna(0,inplace=True)
        test_data[col].fillna(0,inplace=True)

    encode=LabelEncoder()
    for col in categorical_fields:
        data[col]=encode.fit_transform(data[col])
        test_data[col]=encode.fit_transform(test_data[col])
    data['SalePrice'].fillna(0,inplace=True)

pre_process_data()
train_label = data['SalePrice']
train_data = data.drop('Id', axis=1)
train_data = train_data.drop('SalePrice',axis=1)



model= RandomForestClassifier(n_estimators=1000)
# Train the model using the training sets and check score
print ("生成隨機森林...")
model.fit(train_data, train_label)
print ("輸入資料建立模型....")
#Predict Output
print ("輸入test.csv 建立預測")

predicted= model.predict(test_data)

submission = pd.DataFrame({
        "Id": test_data['Id'],
        "SalePrice": predicted
    })
    
submission.to_csv("kaggle.csv", index=False)