from sklearn import linear_model
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

def del_data(data):
    data = data.drop(data[(data.GrLivArea > 4000) & (data.SalePrice < 300000)].index)
    return data


data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data_train = del_data(data_train)


data_X = pd.concat([data_train.drop('SalePrice',1), data_test], ignore_index=True)

def count_missing(data):
    null_cols = data.columns[data.isnull().any(axis=0)]
    X_null = data[null_cols].isnull().sum()
    X_null = X_null.sort_values(ascending=False)
    print(X_null)
    return X_null

def hot_encode_same_att(data_X,attributes):
    
    dfs = [pd.get_dummies(data_X[atts]) for atts in attributes]
    df=dfs[0]
    for frame in dfs[1:]:
        new_att = [att for att in frame.columns if att not in df.columns]
        df = pd.concat([df, frame[new_att]], axis=1)
        for att in frame.columns:
            df[att] = (df[att]+frame[att]).astype(bool).astype(int)
    return df.drop(df.columns[0],1)

def fill_missing(data_X):
    
    catfeats_fillnaNone = ['Alley', 'BsmtCond','BsmtQual','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                           'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC',
                           'Fence', 'MiscFeature']
    data_X.loc[:,catfeats_fillnaNone] = data_X[catfeats_fillnaNone].fillna('None')

    numfeats_fillnazero = ['BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2',
                           'BsmtUnfSF','GarageYrBlt','GarageArea', 'GarageCars']
    data_X.loc[:,numfeats_fillnazero] = data_X[numfeats_fillnazero].fillna(0)


    catfeats_fillnamode = ['Electrical', 'MasVnrType', 'MSZoning', 'Functional', 'Utilities', 'Exterior1st',
                           'Exterior2nd', 'KitchenQual', 'SaleType']
    data_X.loc[:, catfeats_fillnamode] = data_X[catfeats_fillnamode].fillna(data_X[catfeats_fillnamode].mode().iloc[0])
 
    numfeats_fillnamedian = ['MasVnrArea', 'LotFrontage']
 
    data_X.loc[:, numfeats_fillnamedian] = data_X[numfeats_fillnamedian].fillna(data_X[numfeats_fillnamedian].median())

def cat_to_num(data_X):
    
    data_X.Street = data_X.Street.replace({'Pave':2, 'Grvl':1})
    data_X.LotShape = data_X.LotShape.replace({'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1})
    data_X.LandContour = data_X.LandContour.replace({'Lvl':4, 'Bnk':3, 'HLS':2, 'Low':1})
    data_X.LandSlope = data_X.LandSlope.replace({'Gtl':3, 'Mod':2, 'Sev':1})
    data_X.ExterQual = data_X.ExterQual.replace({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})
    data_X.ExterCond = data_X.ExterCond.replace({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})
    data_X.BsmtQual = data_X.BsmtQual.replace({'Ex':6, 'Gd':5, 'TA':4, 'Fa':3, 'Po':2, 'None':1})
    data_X.BsmtCond = data_X.BsmtCond.replace({'Ex':6, 'Gd':5, 'TA':4, 'Fa':3, 'Po':2, 'None':1})
    data_X.BsmtExposure = data_X.BsmtExposure.replace({'Gd':5, 'Av':4, 'Mn':3, 'No':2, 'None':1})
    data_X.HeatingQC = data_X.HeatingQC.replace({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})
    data_X.CentralAir = data_X.CentralAir.replace({'Y':2, 'N':1})
    data_X.KitchenQual = data_X.KitchenQual.replace({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})
    data_X.Functional = data_X.Functional.replace({'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1})
    data_X.FireplaceQu = data_X.FireplaceQu.replace({'Ex':6, 'Gd':5, 'TA':4, 'Fa':3, 'Po':2, 'None':1})
    data_X.GarageType = data_X.GarageType.replace({'2Types':7, 'Attchd':6, 'Basment':5, 'BuiltIn':4, 'CarPort':3, 'Detchd':2, 'None':1})
    data_X.GarageFinish = data_X.GarageFinish.replace({'Fin':4, 'RFn':3, 'Unf':2, 'None':1})
    data_X.GarageQual = data_X.GarageQual.replace({'Ex':6, 'Gd':5, 'TA':4, 'Fa':3, 'Po':2, 'None':1})
    data_X.GarageCond = data_X.GarageCond.replace({'Ex':6, 'Gd':5, 'TA':4, 'Fa':3, 'Po':2, 'None':1})
    data_X.PavedDrive = data_X.PavedDrive.replace({'Y':3, 'P':2, 'N':1})
    
    hot_encoding_atts = ['MSZoning', 'LotConfig', 'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle',
                         'RoofMatl', 'MasVnrType', 'Foundation', 'Heating', 'Electrical',
                         'SaleType', 'SaleCondition']
    for atts in hot_encoding_atts:
        data_X = pd.concat([data_X.drop(atts,1), pd.get_dummies(data_X[atts],
                            drop_first=True)], axis=1)
    
    same_att =[['Condition1','Condition2'],
                     ['Exterior1st','Exterior2nd'], ['BsmtFinType1', 'BsmtFinType2']]
    for pair in same_att:
        data_X = pd.concat([data_X.drop(pair,1),
                            hot_encode_same_att(data_X,pair)], axis=1)
    return data_X

def Feat_Engg(data_X):
    
    drop_list = ['MoSold', 'Id', 'Utilities', 'PoolQC', 'Alley', 'Fence', 'MiscFeature']
    data_X = data_X.drop(drop_list,1)
    
    En = list(data_X[(data_X.GarageType != 'None') & (data_X.GarageYrBlt == 0)].index)
    data_X.GarageYrBlt.iloc[En] = data_X.YearBuilt.iloc[En]
    data_X.GarageFinish.iloc[En] = data_X['GarageFinish'].mode().iloc[0]
    data_X.GarageCars.iloc[En] = 1.0
    data_X.GarageArea.iloc[En] = float(round(data_X['GarageArea'].mean()))
    data_X.GarageQual.iloc[En] = data_X['GarageQual'].mode().iloc[0]
    data_X.GarageCond.iloc[En] = data_X['GarageCond'].mode().iloc[0]
    
    i=data_X[data_X.GarageYrBlt>2020].index[0]
    ar=np.array(data_X.GarageYrBlt)
    ar[i]=data_X.YearBuilt[i]
    data_X['GarageYrBlt']=ar
    
    return data_X

fill_missing(data_X)
data_X = Feat_Engg(data_X)
data_X = cat_to_num(data_X)


train_X = data_X[:-1459]
test_X = data_X[-1459:]
train_Y = np.log(data_train.SalePrice)

ols = linear_model.LinearRegression(fit_intercept = True, n_jobs = -1)

ols.fit(train_X, train_Y)

test_Y = ols.predict(test_X)
test_Y = np.exp(test_Y)
result = pd.DataFrame({'Id': data_test.Id,
                       'SalePrice': test_Y})
result.to_csv('submission.csv', index=False)

scores = cross_val_score(ols, train_X, train_Y, cv=5, scoring = 'neg_mean_squared_error')
scores = np.sqrt(abs(scores))
print("CV score: ", scores.mean())


"""CV SCORE: 0.12043(986)"""









