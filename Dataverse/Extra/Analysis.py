# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:10:56 2019

@author: Loizos
"""

import numpy as np
import pandas as pd
import Import_
import Sampling
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import os
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics
import Utilities

#%% 
# We compute the correlations between the fetures and produce a correlation matrix
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path += '\Results\\'
name = 'adult.data'
dir_p = 'data'
headers = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation',
           'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country',
           'income']

colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'spring', 
             'summer', 'autumn', 'winter', 'cool']
# For more options see https://matplotlib.org/users/colormaps.html

corr_coef = 'P'

data_frame = Import_.import_data(name, dir_p = dir_p, headers = headers, save = True)
data, encoders = Utilities.encode_categorical(data_frame)
corr_ = Utilities.Correlation(data, corr_coef)

# Plot Correlation matrix
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
im, cbar = Utilities.heatmap(corr_, corr_.columns, corr_.columns, cmap=colormaps[0], ax = ax)
texts = Utilities.annotate_heatmap(im, valfmt="{x:.2f}")
fig.tight_layout()
fig.savefig(dir_path+'Initial Correlation Corr Coef_' + corr_coef + '.png')

#%% 
# We replace Husband and Wife with Spooce, in relationship column because,
# there is a strong correlation between sex and relationship. Husband corresponds to Male 
# spouce and Wife to Female. WE are only interested in relationship statues not 
# the gender in the relationship column and spouce is universal, capturing the
# same result with out the correlation. Also Education is correlated with education-num
# which esentially is the numeric form of education, so we also drop Education.

data_frame = data_frame.replace({' Husband':'Spouce', ' Wife':'Spouce'})
del data_frame['education']
data, features = Utilities.encode_categorical(data_frame)
corr_ = Utilities.Correlation(data, corr_coef)

# Create new correlation matrix
fig2 = plt.figure(figsize=(10,10))
ax = fig2.add_subplot(1,1,1)
im, cbar = Utilities.heatmap(corr_, corr_.columns, corr_.columns, cmap=colormaps[0], ax = ax)
texts = Utilities.annotate_heatmap(im, valfmt="{x:.2f}")
fig2.tight_layout()
fig2.savefig(dir_path+'Replaced data Correlation Corr Coef_' + corr_coef + '.png')

#%% 
# Calculate the mean of all continues data. Pandas regognize non-numeric fields
# and skips them. No need for anythink fancy. If there is missing data or
# a row of a continues feature contains non-numeric data, it is ignored.
Means = data_frame.mean(axis = 0)
print('Continues Data Means')
print('--------------------')
print(Means, '\n')
print('--------------------')
print('Targets\' Ratio presence in data', features['income'].classes_[0],
      ':', features['income'].classes_[1])

# Since we have binary classification first class is assign to zero by our encoder
# and second class to 1. The sum of the income column will give the number of 
# the second class.
sums = data.sum(axis = 0)
n = sums['income']
ratio = float(len(data) - n) / float(n)
print('\t', ratio)

if ratio > 1.5 or ratio < 0.67:
    print('It appears we have an imbalanced data set')
print()

#%% 
# We also include in our analysis a base model to compare the rest of the models with.
# For this we will use a logistic regression. But first we need to encode our categorical
# data in a way that does not indruduce spurius distances between variables. For example
# the above methods in the workclass column, Private was given the numeric value of 0
# Self-emp-not-inc the value of 1, Self-emp-inc the value of 2 and so forth. This is wrong 
# as there is no reason why Private should be 'closser' to any other veriable than then the
# rest. This will introduce errors in any kind of regression models, as the weights of veriables
# will be affected by these distances. We can use from sklearn label_binarize, that will
# encode the classes as a one hot encoding, but that will make the logistic regression 
# very ineficient. Instead we can extract the different classes of each column as dummy
# columns (veriables) to the original data frame. Also we want to remove the mean from all of
# our columns and scale the values to have unit variance. We can use sklearn StandardScaler for this.

''' The part below should be substituted by Utiilities preprocess method '''
dummy_data = pd.get_dummies(data_frame).astype('float64')
dummy_data['income'] = dummy_data['income_ >50K']
del dummy_data['income_ <=50K']
del dummy_data['income_ >50K']

indices = np.array(dummy_data.index.values).astype(np.int64)
train = Sampling.k_folds(indices, samples = 10, dir_p = dir_p, save_sample = False)

# Create training/testing data set and training/testing target sets.
# Here we will only validate with one fold and not the full set. 
# But in general when using k-folds we iterate throught all folds
# and use one as validation/testing set and the rest as training.

train_data = dummy_data.loc[train.iloc[:,:-1].values.flatten(),:]
test_data = dummy_data.loc[train.iloc[:,-1].values.flatten(),:]
train_target = train_data.income
test_target = test_data.income
train_data = train_data.drop(columns = ['income'])
test_data = test_data.drop(columns = ['income'])

# Remove mean and set variance of data to one using scaler.
scaler = preprocessing.StandardScaler()
train_data = pd.DataFrame(scaler.fit_transform(train_data), columns = train_data.columns)
test_data = scaler.transform(test_data)
''' end of substitution '''

# Build the model
model = linear_model.LogisticRegression(solver = 'lbfgs')
model.fit(train_data, train_target)

# Make predictions on test data and compute confusion matrix and F1 score.
predict = model.predict(test_data)
conf_matrix = metrics.confusion_matrix(test_target, predict)
f1 = metrics.f1_score(test_target, predict)

# Plot confusion matrix
fig3 = plt.figure(figsize=(6,6))
ax = fig3.add_subplot(1,1,1)
im, cbar = Utilities.heatmap(conf_matrix, features['income'].classes_, 
                   features['income'].classes_, cmap=colormaps[0], ax = ax)
texts = Utilities.annotate_heatmap(im, valfmt="{x:.2f}")

ax.set_title('Confussion Matrix with F1:'+ str("{0:.4f}".format(f1)))
fig3.tight_layout()
fig3.savefig(dir_path + 'Confusion matrix of Logistic Regression Corr Coef_' + 
             corr_coef + '.png')

#%%
''' Test set
name = 'adult.test'

data_frame = Import_.import_data(name, dir_p = dir_p, headers = headers, save = True)
data_frame = data_frame.replace({' Husband':'Spouce', ' Wife':'Spouce'})
del data_frame['education']
data, features = encode_categorical(data_frame)

dummy_data = pd.get_dummies(data_frame)
dummy_data['income'] = dummy_data['income_ >50K.']
del dummy_data['income_ <=50K.']
del dummy_data['income_ >50K.']

test_data = dummy_data
test_target = test_data.income
test_data = test_data.drop(columns = ['income'])
test_data = scaler.transform(test_data)

model.fit(train_data, train_target)

# Make predictions on test data and compute confusion matrix and F1 score.
predict = model.predict(test_data)
conf_matrix = metrics.confusion_matrix(test_target, predict)
f1 = metrics.f1_score(test_target, predict)
print('Confussion Matrix:', '\n', conf_matrix)
print('F1',f1)

# Plot confusion matrix
fig3 = plt.figure(figsize=(8,8))
ax = fig3.add_subplot(1,1,1)
im, cbar = heatmap(conf_matrix, features['income'].classes_, 
                   features['income'].classes_, cmap=colormaps[0], ax = ax)
texts = annotate_heatmap(im, valfmt="{x:.2f}")
ax.set_title('Test set confussion Matrix with F1:'+ str(f1))
fig3.tight_layout()
fig3.savefig(dir_path + 'Confusion matrix of Logistic Regression Corr Coef_' + 
             corr_coef + '.png')
'''
