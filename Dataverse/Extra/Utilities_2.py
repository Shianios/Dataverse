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
import itertools as it

''' All of the function associated with preprocessing should be in a class to
    avoid returning all of those variables'''
# Decorator to convert feeded data to pandas dataframe when needed.
def To_Frame(func):
    def convert(data,*args,**kwargs):
        if not isinstance(data, pd.DataFrame): 
            print('Data is not a pandas DataFrame. Will change to DataFrame')
            if isinstance(data, pd.Series) :
                print('Series detected')
                data = data.to_frame().copy()
            
        else : data = pd.DataFrame(data).copy()
    
        return func(data,**kwargs)
    
    return convert

@To_Frame       
def  Dictionaries(data):
    
    columns_dict = {}

    for column in data.columns:
        if data.dtypes[column] == np.object:
            # Get all unique categorical veriables of all columns 
            values = data[column].unique()
            # sklearn endoder first sort data and then encode. If we do not sort
            # the return dataframe will have incorrect headings.
            values.sort()       
            columns_dict[column] = [x for x in values]
            ''' [x.lower() for x in values] - Add different function to convert all data
                to lower case and remove any white psaces or punctuation marks. '''
    
    return columns_dict

def Encode(data, method = 'Binary'):
    
    mappings = {}
    for key in data.keys():
        if method == 'Binary': encoder = preprocessing.LabelBinarizer()
        elif method == 'Series': encoder = preprocessing.LabelEncoder()
        else:
            print('Encoder Method not valid. Set to Binary')
            encoder = preprocessing.LabelBinarizer()
            method = 'Binary'

        mappings[key] = encoder.fit(data[key])
        
    mappings['method'] = method

    return mappings

@To_Frame
def Fit_Encode(data, method = 'Binary', mappings = None, columns_dict = None):
    
    if not mappings:
        print('No mapps were supplied. Creating maps.')
        if not columns_dict:
            print('No column dictionaries supplied. Creating dics.')
            columns_dict = Dictionaries(data)
        
        mappings = Encode(columns_dict, method)
    
    if not columns_dict:
        print('No column dictionaries supplied. Creating dics.')
        columns_dict = Dictionaries(data)
    
    if not isinstance(data, pd.DataFrame): 
        print('Data is not a pandas DataFrame. Changing to DataFrame')
        if isinstance(data, pd.Series) :
            data = data.to_frame().copy()
            
        else : data = pd.DataFrame(data).copy()
    
    headers = []
    for column in data.columns:
        if data.dtypes[column] == np.object:
            if mappings['method'] == 'Binary' :
                names = ['_' + s for s in columns_dict[column]]
                column_headers = [x+y for x, y in it.product([column], names)]
                headers = headers + column_headers
                
            else : headers = headers + [column]
            
            c_encoding = mappings[column].transform(data[column].values)

            if mappings['method'] == 'Binary' and len(np.squeeze(c_encoding).shape) == 1:
                c_encoding = np.hstack((1 - c_encoding, c_encoding))
            else:
                c_encoding = np.squeeze(c_encoding)
                if len(c_encoding.shape) == 1: c_encoding = c_encoding.reshape(-1,1)
                
            try:
                res = np.append(res, c_encoding, axis = 1)
            except:
                res = c_encoding
            
        else: 
            headers = headers + [column]
            try:
                res = np.append(res, data[column].values.reshape(-1,1), axis = 1)
            except:
                res = data[column].values.reshape(-1,1)

    return pd.DataFrame(res, index = data.index, columns = headers), mappings, columns_dict

@To_Frame
def Remove(df, dictionaries = {}):
    
    data = df.copy()
    for column in data.columns:
        if data.dtypes[column] == np.object:
            values = data[column].unique()
            values = [x.lower() for x in values]
            to_remove = list(set(values) - set(dictionaries[column]))
            for x in to_remove:
                data = data[data[column] != x]
    
    return data

@To_Frame   
def Normalization(data, method = 'l2', scaler = None):
    
    if method in ['l1', 'l2', 'max']:
        scaler = None
        data = pd.DataFrame(preprocessing.normalize(data,
                                        norm = method, axis = 1),
                                        index = data.index,
                                        columns = data.columns)
    
    elif method == 'standard':
        if not scaler : scaler = preprocessing.StandardScaler()
        if data.values.shape[1] == 1:
            data = pd.DataFrame(scaler.fit_transform(data.values.reshape[-1,1]),
                                index = data.index,
                                columns = data.columns)
        else:
            data = pd.DataFrame(scaler.fit_transform(data.values),
                                index = data.index,
                                columns = data.columns)
    
    return data, scaler

''' Need to break to two methods, fisrt sampling (returning data and target sets)
    and second fitting dimensionlity reduction'''
@To_Frame  
def Preprocess(data_frame, target = None, method = 'FAMD', samples = None, 
               mapper = None, num_components = 3, scaler = None,
               encode_method = 'Binary', target_encoder = None,
               data_encoder = None,  data_columns_dict = None, 
               target_column_dict = None, groups = None, normalization = 'l2'):
    
    # If no target supplied get as target the last column of df
    if not target: target = data_frame.columns.values.tolist()[-1] 
    
    ''' TO DO: Fix PCA. '''
    if method == 'PCA': 
        print('Dummy is not functionning proberly.')
        method = 'MFA'
    
    normalization = normalization.lower()
    if normalization not in ['l1', 'l2', 'max', 'standard', None]:
        print('Not a valid normalization method change to None')
        normalization = None
    
    if samples is not None: 
        
        # Sample the data set, Split to training and testing sets.
        train_data = data_frame.loc[samples.iloc[:,:-1].values.flatten(),:]
        test_data = data_frame.loc[samples.iloc[:,-1].values.flatten(),:]
        train_target = train_data[target].copy()
        test_target = test_data[target].copy()
        train_data = train_data.drop(columns = [target])
        test_data = test_data.drop(columns = [target])

        # Encode the data sets
        train_data, data_encoder, data_columns_dict = Fit_Encode(train_data, 
                                                     method = encode_method)
        test_data, _, _ = Fit_Encode(test_data, 
                                     mappings = data_encoder,
                                     columns_dict = data_columns_dict,
                                     method = encode_method)
        #print('Test','\n',train_data.iloc[0])
        train_target, target_encoder, target_column_dict = Fit_Encode(train_target, 
                                                     method = encode_method)
        test_target, _, _ = Fit_Encode(test_target, 
                                       mappings = target_encoder,
                                       columns_dict = target_column_dict,
                                       method = encode_method)

    else: # If no samples are supplied we process the entire data set as a whole.
        test_data = data_frame.copy()
        test_target = test_data[target].copy()
        test_data = test_data.drop(columns = [target]) 
        test_data, test_data_encoder, test_columns_dict = Fit_Encode(test_data, 
                                                  mappings = data_encoder,
                                                  columns_dict = data_columns_dict,
                                                  method = encode_method)
        print('Test Data Encoded')
        test_target, test_target_encoder, _ = Fit_Encode(test_target, 
                                 mappings = target_encoder,
                                 columns_dict = target_column_dict,
                                 method = encode_method)
        print('Test targets encoded')
    
        # Drop the income column from data sets and get normalized vectors
           
    
    if method == 'MFA':
        
        if not groups:
            groups = {}
            for key in data_columns_dict.keys():
                names = ['_' + s for s in data_columns_dict[key]]
                column_headers = [x+y for x, y in it.product([key], names)]
                groups[key] = column_headers
        
        if not mapper: # Create FAMD mapper. 
            print('No mapper found')
            ''' Consider passing **kwargs in Preprocess func. to pass in mappers. '''
            mfa = pr.MFA(
                    groups = groups,
                    n_components = num_components,
                    n_iter=100,
                    #rescale_with_mean = True, # Does not work. Can use sklearn Standard scaller.
                    #rescale_with_std = True,
                    copy=True,
                    check_input=True,
                    engine='auto',
                    random_state=None
                    )
        
        print('Fitting MFA')
        if samples is not None: 

            # Vectors for training/test set
            mapper = mfa.fit(train_data)
            vecs_train = pd.DataFrame(mapper.row_coordinates(train_data))
            vecs_test = pd.DataFrame(mapper.transform(test_data))
            
            vecs_train, scaler = Normalization(vecs_train, normalization, scaler)
            vecs_test, scaler = Normalization(vecs_test, normalization, scaler)
       
            return vecs_train, train_target, vecs_test, test_target, data_columns_dict, target_column_dict, data_encoder, target_encoder, groups, target, mapper, scaler
        
        else:
            # Get the vectors created for the training set and normalise
            vecs_test = pd.DataFrame(mapper.transform(test_data))
            
            vecs_test, scaler = Normalization(vecs_test, normalization, scaler)
        
            ''' Consider returning a single dictionary with all parameters. Each case has 
                different number of returned variables.'''
            
            return vecs_test, test_target, test_data_encoder, test_target_encoder, mapper, target, scaler

    elif method == 'PCA':
        
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
    
            pca_train = mapper.fit(train_data )
            vecs_train = pd.DataFrame(pca_train.row_coordinates(train_data))
            pca_test = mapper.transform(test_data)
            vecs_test = pd.DataFrame(pca_test.row_coordinates(test_data))

            if normalization in ['l1', 'l2', 'max']:
                scaler = None
                vecs_train = pd.DataFrame(preprocessing.normalize(vecs_train,
                                        norm = normalization, axis = 1),
                                        columns = vecs_train.columns)
                
                vecs_test = pd.DataFrame(preprocessing.normalize(vecs_test,
                                    norm = normalization, axis = 1),
                                    columns = vecs_test.columns)
                
            elif normalization == 'standard':
                scaler = preprocessing.StandardScaler()
                vecs_train = pd.DataFrame(scaler.fit_transform(vecs_train), columns = vecs_train.columns)
                vecs_test = pd.DataFrame(scaler.fit_transform(vecs_test), columns = vecs_test.columns)
            
            return vecs_train, train_target, vecs_test, test_target, target_encoders, data_endoder, mapper, target, scaler
        
        else:
            
            test_data, data_endoder = encode_categorical(test_data[target].copy(),
                                                    encode_method = encode_method,
                                                    encoder = data_encoder)
    
            pca_test = mapper.fit(test_data)
            vecs_test = pd.DataFrame(pca_test.row_coordinates(test_data))
            
            if normalization in ['l1', 'l2', 'max']:
                scaler = None
                vecs_test = pd.DataFrame(preprocessing.normalize(vecs_test,
                                    norm = normalization, axis = 1),
                                    columns = vecs_test.columns)
                
            elif normalization == 'standard':
                scaler = preprocessing.StandardScaler()
                vecs_test = pd.DataFrame(scaler.fit_transform(vecs_test), columns = vecs_test.columns)

            return vecs_test, test_target, mapper, target
        
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