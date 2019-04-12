# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 05:33:17 2019

@author: Loizos
"""

import prince as pr
import pandas as pd
import sklearn.preprocessing as preprocessing
import numpy as np
import Import_
import Sampling as sp
import xgboost as xgb
import sklearn.metrics as metrics

# For XGBoost see: https://xgboost.readthedocs.io/en/release_0.72/index.html
# For prince see: https://github.com/kormilitzin/Prince

#%% Similar as in Analysis but for series instead.
def encode_categorical_S(df):
    
    result = df.copy()
    encoders = []
    if result.dtypes == np.object:
        encoders = preprocessing.LabelEncoder()
        result = pd.DataFrame(encoders.fit_transform(result), columns = ['income'])
    return result, encoders

#%% Import data and create sample. 
name = 'adult.csv'
dir_p = 'data'

headers = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation',
           'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country',
           'income']

data_frame = Import_.import_data(name, dir_p = dir_p, headers = headers, save = False)
data_frame = data_frame.replace({' Husband':'Spouce', ' Wife':'Spouce'})
del data_frame['education']
data_frame = data_frame.dropna(axis = 0)

indices = np.array(data_frame.index.values).astype(np.int64)
samples = sp.k_folds(indices, samples = 10, dir_p = dir_p, save_sample = False)
train_data = data_frame.loc[samples.iloc[:,:-1].values.flatten(),:]
test_data = data_frame.loc[samples.iloc[:,-1].values.flatten(),:]

train_data = train_data.drop(columns = ['income'])
test_data = test_data.drop(columns = ['income'])

to_encode = data_frame['income'].copy()
labels, fetures = encode_categorical_S(to_encode)
train_target = pd.DataFrame(labels.loc[train_data.index.values], columns = ['income'])
test_target = pd.DataFrame(labels.loc[test_data.index.values], columns = ['income'])

#%%
# Use FAMD (Factor Analysis for Mixed Data), to reduse the dimensions of the data set
# and convert categorical data to numeric form.
''' Rename test to valid'''
famd = pr.FAMD(
     n_components=5,
     n_iter=10,
     copy=True,
     check_input=True,
     engine='auto',
     random_state=None
 )


famd_train = famd.fit(train_data)
vecs_train = famd_train.row_coordinates(train_data)
famd_test = famd.fit(test_data)
vecs_test = famd_test.row_coordinates(test_data)

scaler = preprocessing.StandardScaler()
vecs_train = pd.DataFrame(scaler.fit_transform(vecs_train), columns = vecs_train.columns)
vecs_test = pd.DataFrame(scaler.transform(vecs_test), columns = vecs_test.columns)

#%% Model
# fit Boosted Tree model on training data
model = xgb.XGBClassifier()
model = model.fit(vecs_train, train_target)

predict = model.predict(test_target)
predictions = [value for value in predict]
conf_matrix = metrics.confusion_matrix(test_target, predict)
f1 = metrics.f1_score(test_target, predict)
print('Confussion Matrix:', '\n', conf_matrix)
print('F1 score',f1)

'''
    !!! NOTE: Even with training set used during predictions, most predictions are 0. Partly
    because of the imbalance of the data set but still F1 is very low, >0.6. Need to investigate.
'''

# Another method to fit a model with XGBoost.
'''
d_train = xgb.DMatrix(vecs_train)
d_test = xgb.DMatrix(vecs_test)
print(d_train)
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
watchlist = [(d_test, 'eval'), (d_train, 'train')]
num_round = 2
model1 = xgb.train(param, d_train, num_round, watchlist)
'''


