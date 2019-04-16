# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 19:49:13 2019

@author: Loizos
"""


import pandas as pd
import numpy as np
import os
''' NOTE 1: We DO NOT sample the entire dataset but rather just the unique indices.
    NOTE 2: We DO NOT save the actual index tables created by sampling, but rather we
            only save the parameters of the sampling meathod and the seed used. This way
            we save space and the results are easily reproduced.
'''
#%%
def bootstrapping(indices, samples = 1, train_size = None, test_size = None,
                  seed = None, save_sample = False, dir_p = 'data'):
    
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    if train_size == None: train_size = len(indices)
    if seed == None : seed = np.floor(np.random.uniform(0, 4294967295)).astype(np.int64)
    np.random.seed(seed)
    
    for i in range(samples):
        s_name = 'S' + str(i)
        # Randomly choose n indices with replacement n=train_size. Create also new list
        # with the indeces that were not selected, to be used in the creation of test sample.
        train_ind = np.random.choice(indices, size = train_size, replace = True)
        reduced_indices = np.delete(indices, train_ind)

        if test_size == None: test_size = len(reduced_indices)
        test_ind = np.random.choice(reduced_indices, size = test_size, replace = True)
        
        # We return the samples as pandas series.
        df_tr = pd.DataFrame(train_ind, columns = [s_name])
        df_ts = pd.DataFrame(test_ind, columns = [s_name])
        df_train = pd.concat([df_train,df_tr], axis = 1)
        df_test = pd.concat([df_test,df_ts], axis = 1)
        
    if save_sample == True:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if dir_p[0] == '/' or dir_p[0] == '\\':
            if dir_p[-1] == '/' or dir_p[-1] == '\\' : dir_path += dir_p
            else: dir_path += dir_p + '/'
        else:
            if dir_p[-1] == '/' or dir_p[-1] == '\\' : dir_path += '\\' + dir_p
            else: dir_path += '\\' + dir_p + '\\'
        file_name = dir_path + 'B_samples.csv'

        try:
            data_frame = pd.read_csv(file_name)
            index = len(data_frame)
            params = {'seed':seed, 'samples':samples, 'train_size': train_size, 'test_size': test_size}
            res = pd.DataFrame(data = params, index = [index])
            with open(file_name, 'a') as file:
                res.to_csv(file, header = False)
        except:
            params = {'seed':seed, 'samples':samples, 'train_size': train_size, 'test_size': test_size}
            res = pd.DataFrame(data = params, index = [0])
            res.to_csv(file_name)
            
    return df_train, df_test

#%%
def k_folds(indices, samples = 1, seed = None, save_sample = True, dir_p = 'data'):
    
    size = int(float(len(indices))/float(samples))
    df_samples = pd.DataFrame()
    if seed == None : seed = np.floor(np.random.uniform(0, 4294967295)).astype(np.int64)
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    # In k-fold first we shuffle the data and then split it into k groups, (k = samples).
    # The sample size is the total number of entries devided by k.
    for i in range (samples):
        k_name = 'K' + str(i+1)
        df = pd.DataFrame(indices[i*size:(i+1)*size], columns = [k_name])
        df_samples = pd.concat([df_samples,df], axis = 1)
        
    if save_sample == True:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if dir_p[0] == '/' or dir_p[0] == '\\':
            if dir_p[-1] == '/' or dir_p[-1] == '\\' : dir_path += dir_p
            else: dir_path += dir_p + '/'
        else:
            if dir_p[-1] == '/' or dir_p[-1] == '\\' : dir_path += '\\' + dir_p
            else: dir_path += '\\' + dir_p + '\\'
        file_name = dir_path + 'K_samples.csv'

        try:
            data_frame = pd.read_csv(file_name)
            index = len(data_frame)
            params = {'seed':seed, 'samples':samples}
            res = pd.DataFrame(data = params, index = [index])
            with open(file_name, 'a') as file:
                res.to_csv(file, header = False)
        except:
            params = {'seed':seed, 'samples':samples}
            res = pd.DataFrame(data = params, index = [0])
            res.to_csv(file_name)
    
    return df_samples



#%% Test
'''
import Import_
 
name = 'adult.data'
dir_p = 'data'
#name = 'adult.csv'
headers = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation',
           'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country',
           'income']  

data_frame = Import_.import_data(name, dir_p = dir_p, headers = headers, save = False)
indices = np.array(data_frame.index.values).astype(np.int64)

switch = 'b'
if switch == 'b':
    train, test = bootstrapping(indices, samples = 10, train_size = 100, test_size = 50, dir_p = dir_p, save_sample = True)
else:
    train, _ = k_folds(indices, samples = 10, dir_p = dir_p, save_sample = True)
print(train)
print(len(train))
'''
