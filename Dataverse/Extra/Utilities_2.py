# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:27:56 2019

@author: Loizos
"""

import numpy as np
import pandas as pd
import prince as pr
import sklearn.preprocessing as preprocessing
import matplotlib
import matplotlib.pyplot as plt

def encode_categorical(df, encoders = {}):
    
    if not isinstance(encoders, dict): 
        print('Encoders are not a dict. Will change to empty dict')
        encoders = {}
    if not isinstance(df, pd.DataFrame): 
        print('Data is not a pandas DataFrame. Will dhange to DataFrame')
        result = pd.DataFrame(df)
    else:
        result = df.copy()
        
    for column in result.columns:
        if result.dtypes[column] == np.object:
            try :
                result[column] = encoders[column].fit_transform(result[column])
            except:
                encoders[column] = preprocessing.LabelEncoder()
                result[column] = encoders[column].fit_transform(result[column])
                
    return result, encoders

def Preprocess(data_frame, target = None, method = 'FAMD', samples = None, 
               mapper = None, num_components = 3, scaler = None):
    
    # If no target supplied get as target the last column of df
    if not target: target = data_frame.columns.values.tolist()[-1] 
    
    ''' See below for this issue, there is a problem for now with Dummy '''
    if method == 'Dummy': 
        print('Dummy is not functionning proberly.')
        method = 'PCA'
    
    
    if samples is not None: 
        ''' Use different encoding method. sklearn LabelBinarizer or OneHotEncoder'''
        
        # Sample the data set, Split to training and testing sets.
        train_data = data_frame.loc[samples.iloc[:,:-1].values.flatten(),:]
        test_data = data_frame.loc[samples.iloc[:,-1].values.flatten(),:]

        # Create taining labels. This will give us a one-hot encoding for each class.
        train_target = pd.get_dummies(train_data[target]).astype('float64')
    
        # Create testing labels
        test_target = pd.get_dummies(test_data[target]).astype('float64')
    
        # Drop the income column from data sets.
        train_data = train_data.drop(columns = [target])
        test_data = test_data.drop(columns = [target])
        
    else: # If no samples are supplied we process the entire data set as a whole.
        test_data = data_frame.copy()
        test_target = pd.get_dummies(test_data[target]).astype('float64')
    
        # Drop the income column from data sets and get normalized vectors
        test_data = test_data.drop(columns = [target])
        
    
    if method == 'FAMD':
        
        if not mapper: # Create FAMD mapper. 
            ''' Consider passing **kwargs in Preprocess func. to pass in mappers. '''
            mapper = pr.FAMD(
                n_components = num_components,
                n_iter=100,
                #rescale_with_mean = True,
                #rescale_with_std = True,
                copy=True,
                check_input=True,
                engine='auto',
                random_state=None
            )
        
          
        # Get the vectors created for the training set and normalise
        famd_test = mapper.fit(test_data)
        vecs_test = pd.DataFrame(famd_test.row_coordinates(test_data))
        vecs_test = pd.DataFrame(preprocessing.normalize(vecs_test, norm = 'l2', axis = 1), columns = vecs_test.columns)
        
        if samples is not None: 
            # Vectors for training set
            famd_train = mapper.fit(train_data)
            vecs_train = pd.DataFrame(famd_train.row_coordinates(train_data))
            vecs_train = pd.DataFrame(preprocessing.normalize(vecs_train, norm = 'l2', axis = 1), columns = vecs_train.columns)
       
            return vecs_train, train_target, vecs_test, test_target, mapper, target
        
        ''' Consider returning a single dictionary. Each case has 
            different number of returned variables.'''
            
        return vecs_test, test_target, mapper, target

    elif method == 'PCA': # PCA only works with numerical data. See below how we convert non numeric.
        
        if not mapper:
            
            mapper = pr.PCA(
                n_components = num_components,
                n_iter = 100,
                rescale_with_mean = True,
                rescale_with_std = True,
                copy = True,
                check_input = True,
                engine = 'auto',
                random_state = None
            )
        
        if samples is not None:
            
            train_data =  pd.get_dummies(train_data).astype('float64')
            test_data = pd.get_dummies(test_data).astype('float64')
    
            pca_train = mapper.fit(train_data )
            vecs_train = pd.DataFrame(pca_train.row_coordinates(train_data))
            pca_test = mapper.fit(test_data)
            vecs_test = pd.DataFrame(pca_test.row_coordinates(test_data))

            #vecs_train = pd.DataFrame(preprocessing.normalize(vecs_train, norm = 'l2', axis = 1), columns = vecs_train.columns)
            #vecs_test = pd.DataFrame(preprocessing.normalize(vecs_test, norm = 'l2', axis = 1), columns = vecs_test.columns)
            
            return vecs_train, train_target, vecs_test, test_target, mapper, target
        
        else:
            
            test_data = pd.get_dummies(test_data).astype('float64')
    
            pca_test = mapper.fit(test_data)
            vecs_test = pd.DataFrame(pca_test.row_coordinates(test_data))
            #vecs_test = pd.DataFrame(preprocessing.normalize(vecs_test, norm = 'l2', axis = 1), columns = vecs_test.columns)

            return vecs_test, test_target, mapper, target
        
    elif method == 'Dummy':

        labels = pd.get_dummies(data_frame[target]).astype('float64')
        del data_frame[target]
        data_frame = pd.get_dummies(data_frame).astype('float64')
        
        if samples is not None:
            train_data = pd.get_dummies(train_data).astype('float64')
            test_data = pd.get_dummies(test_data).astype('float64')
            
            if not scaler:
                scaler = preprocessing.StandardScaler()
                vecs_train = pd.DataFrame(scaler.fit_transform(train_data), columns = train_data.columns)
                vecs_test = pd.DataFrame(scaler.transform(test_data), columns = test_data.columns)
            else: 
                '''
                    !!! NOTE: Need to fix. If scaler is supplied but fitted to different dimensional
                    data, it cannot be used and returns error. If test data do not contain
                    a classes of any feature or contains new classes, the dimensions of
                    the dummy data frame will be different. Also this issue will create problems
                    with Tensorflow's placeholders.
                '''
                
                try:
                    vecs_train = pd.DataFrame(scaler.transform(train_data), columns = train_data.columns)
                    vecs_test = pd.DataFrame(scaler.transform(test_data), columns = test_data.columns)
                except:
                    vecs_train = pd.DataFrame(scaler.fit_transform(train_data), columns = train_data.columns)
                    vecs_test = pd.DataFrame(scaler.transform(test_data), columns = test_data.columns)
       
            return vecs_train, train_target, vecs_test, test_target, scaler, target
        
        else:
            test_data = data_frame.copy()
            test_target = labels
            
            if not scaler:
                scaler = preprocessing.StandardScaler()
                vecs_test = pd.DataFrame(scaler.fit_transform(test_data), columns = train_data.columns)
            else:
                try:
                    vecs_test = pd.DataFrame(scaler.transform(test_data), columns = test_data.columns)
                except:
                    vecs_test = pd.DataFrame(scaler.fit_transform(test_data), columns = test_data.columns)
            
            return vecs_test, test_target, scaler, target

def Correlation(data, method = 'P'):
    
    if method == 'P': corr_ = data.corr(method='pearson')
    elif method == 'S': corr_ = data.corr(method='spearman')
    else : corr_ = data.corr(method='kendall')
    
    return corr_

# The following 2 methods were slightly modified from 
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

def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=["black", "blue"],
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
            color_code =  lambda x: 0 if (x < threshold) else 1
            kw.update(color=textcolors[color_code(im.norm(data[i, j]))])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts