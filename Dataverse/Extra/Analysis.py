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
import matplotlib
import matplotlib.pyplot as plt
import os
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics

#%% 
''' 
    The 4 functions need to be placed in diferent file to be callables from other modules.
'''
def encode_categorical(df):
    
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders

def Correlation(data, method = 'P'):
    
    if method == 'P': corr_ = data.corr(method='pearson')
    elif method == 'S': corr_ = data.corr(method='spearman')
    else : corr_ = data.corr(method='kendall')
    
    return corr_

# The following 2 methods were taken from 
# https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on bottom.
    ax.tick_params(top=True, bottom=False,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=75, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=["black", "white"],
                     threshold=None, **textkw):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
#%% 
# We compute the correlations between the fetures and produce a correlation matrix
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path += '\Results\\'
name = 'adult.data'
dir_p = 'data'
headers = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation',
           'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country',
           'income']
corr_coef = 'P'

data_frame = Import_.import_data(name, dir_p = dir_p, headers = headers, save = True)
data, _ = encode_categorical(data_frame)
corr_ = Correlation(data, corr_coef)


colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'spring', 
             'summer', 'autumn', 'winter', 'cool']
# For more options see https://matplotlib.org/users/colormaps.html
# Plot Correlation matrix
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
im, cbar = heatmap(corr_, corr_.columns, corr_.columns, cmap=colormaps[0], ax = ax)
texts = annotate_heatmap(im, valfmt="{x:.2f}")
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
data, features = encode_categorical(data_frame)
corr_ = Correlation(data, corr_coef)

# Create new correlation matrix
fig2 = plt.figure(figsize=(10,10))
ax = fig2.add_subplot(1,1,1)
im, cbar = heatmap(corr_, corr_.columns, corr_.columns, cmap=colormaps[0], ax = ax)
texts = annotate_heatmap(im, valfmt="{x:.2f}")
fig2.tight_layout()
fig2.savefig(dir_path+'Replaced data Correlation Corr Coef_' + corr_coef + '.png')

#%% 
# Calculate the mean of all continues data. Pandas regognize non-numeric fields
# and skips them. No need for anythink fancy. If there is missing data or
# a row of a continues feature contains non-numeric data, it is ignored.
Means = data_frame.mean(axis = 0)
print('Continues Data Meanns')
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

dummy_data = pd.get_dummies(data_frame)
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

# Build the model
model = linear_model.LogisticRegression()
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
ax.set_title('Confussion Matrix with F1:'+ str(f1))
fig3.tight_layout()
fig3.savefig(dir_path + 'Confusion matrix of Logistic Regression Corr Coef_' + 
             corr_coef + '.png')




