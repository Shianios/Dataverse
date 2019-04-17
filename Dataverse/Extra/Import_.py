# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:26:01 2019

@author: Loizos
"""

import pandas as pd
import re
import os 

'''
    TO DO: Entries containe white spaces. Need to remove to avoid errors.
'''
def import_data(file_name = None, dir_p = '/data/', headers = [], save = False):
#%% Validations for directory path and file name
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if dir_p[0] == '/' or dir_p[0] == '\\':
        if dir_p[-1] == '/' or dir_p[-1] == '\\' : dir_path += dir_p
        else: dir_path += dir_p + '/'
    else:
        if dir_p[-1] == '/' or dir_p[-1] == '\\' : dir_path += '\\' + dir_p
        else: dir_path += '\\' + dir_p + '\\'

    while not os.path.isdir(dir_path):
        dir_path = input('Default directory does not exist. Insert new directory:')
    while type(file_name).__name__ != 'str':
        file_name = input('File name is not a string insert new name:')
    file_path = dir_path + file_name
    while not (os.path.isfile(file_path)):
        file_name = input("File does not exist. Insert new name. To insert full path use 'p-C:...\..\..' :")
        if file_name[:2] == 'p-':
            file_path = file_name[2:]
        else:
            file_path = dir_path + file_name
    
#%% 
    try :
        try:
            data_frame = pd.read_csv(file_path, header = None, sep = ',',index_col = False)
            data_frame.columns = headers
            # In case the header is included in the table we remove the first row and re-indexing
            if data_frame.loc[0, headers[0]] == headers[0]:
                data_frame = data_frame.drop([0],axis=0)
                data_frame.index = range(data_frame.shape[0])
        except:
            data_frame = pd.read_csv(file_path, header = 0, sep = ',',index_col = 0)
            
    except :
        print('Error reading csv. Read as txt instead')
        with open(file_path, 'r') as d_file:
            lines = d_file.readlines()
        d_file.close()
        
        # If headers are not supplied we get the columns' names from the first row of file
        if not headers:
            delim_pos = [m.start() for m in re.finditer(',', lines[0])] 
            delim_pos = [-1] + delim_pos + [len(lines[0])-1]
            headers = []
            for i in range (1,len(delim_pos)):
                headers.append(lines[0][delim_pos[i-1]+1:delim_pos[i]])

            data_frame = pd.DataFrame(columns=headers)
            line1 = 1
        else: 
            data_frame = pd.DataFrame(columns=headers)
            line1 = 0

        for i in range (line1,len(lines[:10])):
            delim_pos = [m.start() for m in re.finditer(',', lines[i])]
            elements = []

            if len(delim_pos)+1 == len(headers):
                delim_pos = [-1] + delim_pos
                for j in range(1,len(delim_pos)):
                    el = lines[i][delim_pos[j-1]+1:delim_pos[j]]
                    elements.append(el)  
                elements.append(lines[i][delim_pos[j]+1:-2])
                
                df = pd.DataFrame([elements], columns=headers, index = [i])
                data_frame = data_frame.append(df)
                del df
            else: print('Missing data in line', i)
        d_file.close()

    if save == True:
        # We save the table as csv purely for compactness and speed when reuploading the table.
        name = file_name[:file_name.find('.')]
        data_frame.to_csv(dir_path + name + '.csv', sep = ',')  
        
    print(file_name,'imported')
    return data_frame
                    
#%% Test
'''
name = 'adult.data'
#name = 'adult.csv'
headers = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation',
           'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country',
           'income']  
    
df = import_data(name, dir_p = 'data', headers = headers, save = True)
h = list(df.columns)
ind = df.index.tolist()
print(df)
'''
