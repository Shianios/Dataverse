# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 05:13:03 2019

@author: Loizos
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface
from scipy.optimize import basinhopping
import sklearn.metrics as metrics
import os
import time
import Import_
import Sampling as sp
#import Utilities
import Utilities_2 as Utilities

tf.reset_default_graph()

# To be used to save the network graph
dir_path = os.path.dirname(os.path.realpath(__file__))
tf_path = dir_path + '\logs'
    
#%% Import data and create sample. 
name = 'adult.data'
dir_p = 'data'

headers = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation',
           'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country',
           'income']

data_frame = Import_.import_data(name, dir_p = dir_p, headers = headers, save = False)
data_frame = data_frame.replace({' Husband':'Spouce', ' Wife':'Spouce'})
del data_frame['education']
data_frame = data_frame.dropna(axis = 0)

condition = data_frame.apply(lambda x: ' ?' in x.values, axis = 1)
ind_to_replace = condition.index[condition].tolist()
data_frame = data_frame.drop(index = ind_to_replace)
indices = np.array(data_frame.index.values).astype(np.int64)
samples = sp.k_folds(indices, samples = 10, dir_p = dir_p, save_sample = False)

preprocess_methods = ['MFA', 'PCA']
pre_meth = preprocess_methods[0]
components = 14

vecs_train, train_target, vecs_test, test_target, data_columns_dict, target_column_dict, data_encoder, target_encoder, groups, target, mapper, scaler = Utilities.Preprocess(
                                                                        data_frame, 
                                                                        method = pre_meth, 
                                                                        samples = samples, 
                                                                        mapper = None, 
                                                                        num_components = components, 
                                                                        scaler = None)

#%% Network and optimization parameters
num_input = vecs_train.shape[1]
num_output = 2
Epochs = 10
display_step = 10 #max(int(Epochs / 10),1)
num_hidden = [7,14]                     # For multidimensional layers pass tuples (Not possible in this ver.)
drop_rate = 0.25                        # Drop rate of dropout layers
learning_rate = 0.1                     # To use in GD
opt = 'BH'                              # Choose between gradient discent 'GD' or BasinHopping 'BH' for optimizaton.
opt_name = 'L-BFGS-B'                   # Optimizer in the second stage of BH
maxiter = 20                            # For optimizer in BH
niter = 10                              # BH iterations
threshold = 0.1                         # Return label class if its prob is greatter than threshold.

np.random.seed(7919)
tf.random.set_random_seed(7919)     
''' Fix random seed only during testing and parameter tunning. We use the 1000th prime number ''' 

#%% mlp forward ops and parameters

def multi_layer_perceptron(x, weights, biases):

    hidden_layer1 = tf.add(tf.matmul(x, weights['w_h1']), biases['b_h1'])
    hidden_layer1 = tf.nn.sigmoid(hidden_layer1)
    hidden_layer2 = tf.add(tf.matmul(hidden_layer1, weights['w_h2']), biases['b_h2'])
    hidden_layer2 = tf.nn.relu(hidden_layer2)

    drop_out = tf.layers.dropout(hidden_layer2, rate = drop_rate)
    
    # We do not use activation for out layer, as it will be included in the optimization
    # for efficiency. During predictions the output should be passed from the act. func.
    out_layer = tf.add(tf.matmul(drop_out , weights['w_out']), biases['b_out'])

    return out_layer

# weights and biases
weights = {
    'w_h1' : tf.Variable(tf.random.uniform([num_input, num_hidden[0]],
                                    minval = -1.,maxval = 1., dtype=tf.float64), ),

    'w_h2' : tf.Variable(tf.random.uniform([num_hidden[0], num_hidden[1]],
                                    minval = -1.,maxval = 1., dtype=tf.float64), ),

    'w_out': tf.Variable(tf.random.uniform([num_hidden[1], num_output],
                                    minval = -1.,maxval = 1., dtype=tf.float64))
}
biases = {
    'b_h1' : tf.Variable(tf.random.uniform([num_hidden[0]],minval = -0.1,
                                           maxval = 0.1, dtype=tf.float64)),
    
    'b_h2' : tf.Variable(tf.random.uniform([num_hidden[1]],minval = -0.1,
                                           maxval = 0.1, dtype=tf.float64)),

    'b_out': tf.Variable(tf.random.uniform([num_output],minval = -0.1,
                                           maxval = 0.1, dtype=tf.float64))
}

X = tf.placeholder(tf.float64, [None, num_input])     # training data
Y = tf.placeholder(tf.float64, [None, num_output])    # labels
x = np.array(vecs_train.values).astype(np.float64)
y = np.array(train_target.values).astype(np.float64)

model = multi_layer_perceptron(X, weights, biases)

#%% Training
'''
For cost functions can try
- softmax cross entropy  (softCE)
- sigmoid cross entropy  (sigCE)
- mean square error      (MSE)
- mean abs error         (MAE)
- Shannon entropy        (SE)
- Shannon with different 
  kernels other than Gaussioan

For optimization try
- gradient descent        (GD)
- from scipy basinhopping (BH)
'''
with tf.Session() as sess:
    writer = tf.summary.FileWriter(tf_path, sess.graph)
    loss_fun = 'softmax cross entropy'
    loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    #loss_func = tf.reduce_mean(tf.losses.mean_squared_error(Y,tf.nn.softmax(model)))
    
    if opt == 'GD':
        opt_name = None
        maxiter = None
        niter = None

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_func)
        tf.global_variables_initializer().run()
        ini_loss = np.around(sess.run(loss_func, feed_dict={X: x, Y: y}), decimals = 5)
        print('Initial Loss:', ini_loss)
        
        start_time = time.time()
        for k in range(Epochs):
            tmp_cost, _ = sess.run([loss_func, optimizer], feed_dict={X: x, Y: y})
            if k % display_step == 0.:
                print('---------- Epoch', k+1, '----------')
                print('Loss:', tmp_cost)
                
    elif opt == 'BH':
        learning_rate = None
        # To test Scipy Optimizer Interface use first option else use second in optimizer.
        # method='L-BFGS-B', options={'maxiter': 2}
        # method = basinhopping, options = {'min_options':{'minimizer_kwargs' : minimizer_kwargs, 'niter' : niter}}
        minimizer_kwargs = {"method": opt_name, 'options':{'maxiter': maxiter}}
        optimizer = ScipyOptimizerInterface(loss_func, method = basinhopping, 
            options = {'bh_options':{'minimizer_kwargs' : minimizer_kwargs, 'niter' : niter}})

        tf.global_variables_initializer().run()
        ini_loss = np.around(sess.run(loss_func, feed_dict={X: x, Y: y}), decimals = 5)
        print('Initial Loss:', ini_loss)

        start_time = time.time()
        for k in range(Epochs):
            print('---------- Epoch', k+1, '----------')
            optimizer.minimize(sess, feed_dict={X: x, Y: y})
            #if k % display_step == 0.:
                #print('Loss:', sess.run(loss_func, feed_dict={X: x, Y: y}))
                
    end_time = time.time()   
    fin_loss = np.around(sess.run(loss_func, feed_dict={X: x, Y: y}), decimals = 5)
    print('Final loss:', fin_loss, '\n')

#%%    
    # Feed the testing data set and predict the class. We also capture the probability
    # by which the network outputted the class.
    x = np.array(vecs_test.values).astype(np.float64)
    predict, max_val = np.array(sess.run([tf.argmax(tf.nn.softmax(model), axis = 1), 
                                tf.reduce_max(tf.nn.softmax(model), axis = 1)], feed_dict={X:x}))     
    
    # The [1,0] vector coresponts to 0 index returned by argmax, whereas [0,1] to index1.
    # If the index returned is 0 that is equivalent to have the label return as 1, so we use
    # modular addition by 1 with mod 2.
    predict = (predict.astype(np.int32) +1) % 2

    # If the max value is less than for example 10% we, interpret that the network couldn 't
    # identify the input. We do this in order to be aple to handle missing data.
    for i in range(len(max_val)):
        if max_val[i] < threshold: predict[i] = -1

    test_labels = np.array(test_target.iloc[:,0]).astype(np.int32)

    conf_matrix = metrics.confusion_matrix(test_labels, predict)
    f1 = metrics.f1_score(test_labels, predict)
    print('Confussion Matrix for validation set:', '\n', conf_matrix)
    print('Validation set\'s F1 score:', f1, '\n')
    
#%% adalt.test Data set
    name = 'adult.test'

    test_frame = Import_.import_data(name, dir_p = dir_p, headers = headers, save = False)
    test_frame = test_frame.replace({' Husband':'Spouce', ' Wife':'Spouce'})
    test_frame = test_frame.replace({' <=50K.':' <=50K', ' >50K.':' >50K'})
    del test_frame['education']
    test_frame = test_frame.dropna(axis = 0)
    
    # Remove None that were replaced with ?.
    condition = test_frame.apply(lambda x: ' ?' in x.values, axis = 1)
    ind_to_replace = condition.index[condition].tolist()
    test_frame = test_frame.drop(index = ind_to_replace)
    
    vecs_test, test_target, test_data_encoder, test_target_encoder, mapper, target, scaler = Utilities.Preprocess(
                                            test_frame, 
                                            method = pre_meth, 
                                            samples = None,
                                            data_columns_dict = data_columns_dict,
                                            target_column_dict = target_column_dict,
                                            data_encoder = data_encoder,
                                            target_encoder = target_encoder,
                                            groups = groups,
                                            mapper = mapper, 
                                            scaler = scaler,
                                            target = target)
    
    x = np.array(vecs_test.values).astype(np.float64)
    predict, max_val = np.array(sess.run([tf.argmax(tf.nn.softmax(model), axis = 1), 
                                tf.reduce_max(tf.nn.softmax(model), axis = 1)], feed_dict={X:x}))

    predict = (predict.astype(np.int32) + 1) % 2

    # If the max value is less than for example 10% we, interpret that the network couldn 't
    # identify the input. We do this in order to be aple to handle missing data.
    for i in range(len(max_val)):
        if max_val[i] < threshold: predict[i] = -1

    test_labels = np.array(test_target.iloc[:,0]).astype(np.int32)

    conf_matrix = metrics.confusion_matrix(test_labels, predict)
    f1 = metrics.f1_score(test_labels, predict)
    print('Confussion Matrix for Test set:', '\n', conf_matrix)
    print('Test set\'s F1 score:',f1)
     