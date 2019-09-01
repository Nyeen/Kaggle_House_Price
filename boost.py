import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost
from sklearn.model_selection import cross_val_score

def del_data(data):
    data = data.drop(data[(data.GrLivArea > 4000) & (data.SalePrice < 300000)].index)
    return data

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train = del_data(df_train)

data_X = pd.concat([df_train.drop('SalePrice',1), df_test], ignore_index=True)

def Feat_Engg(data_X):
    
    data_X['PoolArea']=(data_X['PoolArea']!=0).astype(int)
    drop_list = ['MoSold', 'Id', 'Utilities', 'PoolQC']
    data_X = data_X.drop(drop_list,1)
    
    return data_X

def cat_to_num(data_X):
    cols = ( 
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'FireplaceQu',
            'Alley', 'Fence', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'ExterQual',
            'ExterCond','HeatingQC','KitchenQual', 'Functional', 'LandContour',
            'LandSlope', 'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass',
            'OverallCond', 'YrSold', 'MSZoning', 'LotConfig', 'Neighborhood', 'BldgType',
            'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'Foundation', 'Heating',
            'Electrical', 'MiscFeature', 'SaleType', 'SaleCondition', 'Condition1',
            'Condition2', 'Exterior1st', 'Exterior2nd', 'BsmtFinType1', 'BsmtFinType2',
            'PoolQC', 'Utilities'
            )
    for i in cols:
        lbl = LabelEncoder() 
        lbl.fit(list(data_X[i].values)) 
        data_X[i] = lbl.transform(list(data_X[i].values))
    
    return data_X

data_X = cat_to_num(data_X)

data_X = Feat_Engg(data_X)

train_X = data_X[:-1459]
test_X = data_X[-1459:]
train_Y = np.log(df_train.SalePrice)

xgb = xgboost.XGBRegressor(colsample_bytree=0.25, subsample=0.5,
                             learning_rate=0.025, max_depth=2,
                             min_child_weight=1, n_estimators=2600,
                             reg_alpha=0.19, reg_lambda=0.46, gamma= 0,
                             random_state = 7, nthread = -1, max_delta_step = 1)


xgb.fit(train_X, train_Y)
test_Y = xgb.predict(test_X)
test_Y = np.exp(test_Y)
result = pd.DataFrame({'Id': df_test.Id,
                       'SalePrice': test_Y})
result.to_csv('submission.csv', index=False)

scores = cross_val_score(xgb, train_X, train_Y, cv=5, scoring = 'neg_mean_squared_error')
scores = np.sqrt(abs(scores))
print("CV score: ", scores.mean())


"""CV SCORE: 0.11001(63)"""




