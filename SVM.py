import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def del_data(data):
    data = data.drop(data[(data.GrLivArea > 4000) & (data.SalePrice < 300000)].index)
    return data

##
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data_train = del_data(data_train)

data_X = pd.concat([data_train.drop('SalePrice',1), data_test], ignore_index=True)

def fill_missing(data_X):
    
    catfeats_fillnaNone = ['Alley', 'BsmtCond','BsmtQual','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                           'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC',
                           'Fence', 'MiscFeature']
    data_X.loc[:,catfeats_fillnaNone] = data_X[catfeats_fillnaNone].fillna('None')

    numfeats_fillnazero = ['BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2',
                           'BsmtUnfSF','GarageArea', 'GarageCars']
    data_X.loc[:,numfeats_fillnazero] = data_X[numfeats_fillnazero].fillna(0)




    data_X.loc[:,'GarageYrBlt'] = data_X['GarageYrBlt'].fillna(data_X.YearBuilt)




    catfeats_fillnamode = ['Electrical', 'MasVnrType', 'MSZoning', 'Functional', 'Utilities', 'Exterior1st',
                           'Exterior2nd', 'KitchenQual', 'SaleType']
    data_X.loc[:, catfeats_fillnamode] = data_X[catfeats_fillnamode].fillna(data_X[catfeats_fillnamode].mode().iloc[0])
 
    numfeats_fillnamedian = ['MasVnrArea', 'LotFrontage']
 
    data_X.loc[:, numfeats_fillnamedian] = data_X[numfeats_fillnamedian].fillna(data_X[numfeats_fillnamedian].median())

def Feat_Engg(data_X):
    
    drop_list = ['MoSold', 'Id', 'Utilities', 'PoolQC', 'Alley', 'Fence', 'MiscFeature']
    data_X = data_X.drop(drop_list,1)

    data_X['Old'] = data_X['YrSold']-data_X['YearBuilt']; data_X.drop('YearBuilt',1)
    
    return data_X

fill_missing(data_X)
data_X = Feat_Engg(data_X)

encoder = ce.BackwardDifferenceEncoder(cols=['MSZoning', 'Street', 'LotShape', 'LandContour',
                                             'LotConfig', 'LandSlope', 'Neighborhood',
                                             'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
                                             'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                                             'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                                             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                                             'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',
                                             'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
                                             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                                             'PavedDrive', 'SaleType', 'SaleCondition'])

data_X = encoder.fit_transform(data_X)
data_X = data_X.drop('intercept',1)

min_max_scaler = preprocessing.MinMaxScaler()
data_X = pd.DataFrame(min_max_scaler.fit_transform(data_X), columns = data_X.columns)


train_X = data_X[:-1459]
test_X = data_X[-1459:]
train_Y = np.log(data_train.SalePrice)

machine = SVR(C=7, epsilon=0.01, tol = 0.0001, kernel='poly',
              coef0 = 1.7, degree = 2)
#machine = RandomForestRegressor(max_depth=20)
machine.fit(train_X, train_Y)
test_Y = machine.predict(test_X)
test_Y = np.exp(test_Y)
result = pd.DataFrame({'Id': data_test.Id,
                       'SalePrice': test_Y})
result.to_csv('submission.csv', index=False)


scores = cross_val_score(machine, train_X, train_Y, cv=5, scoring = 'neg_mean_squared_error')
scores = np.sqrt(abs(scores))
print("CV score: ", scores.mean())




