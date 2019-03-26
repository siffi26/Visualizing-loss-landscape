# -*- coding: utf-8 -*-

# data taken from https://archive.ics.uci.edu/ml/datasets/seismic-bumps

# The data describe the problem of high energy (higher than 10^4 J) seismic bumps forecasting in a coal 
# mine. Data come from two of longwalls located in a Polish coal mine.
# Instances: 2584 
# Attributes: 18 + class
# Class distribution: 
#     "hazardous state" (class 1)    :  170  (6.6%)
#     "non-hazardous state" (class 0): 2414 (93.4%)
#
# Attribute information:
#  1. seismic: result of shift seismic hazard assessment in the mine working obtained by the seismic 
# method (a - lack of hazard, b - low hazard, c - high hazard, d - danger state);
#  2. seismoacoustic: result of shift seismic hazard assessment in the mine working obtained by the 
# seismoacoustic method;
#  3. shift: information about type of a shift (W - coal-getting, N -preparation shift);
#  4. genergy: seismic energy recorded within previous shift by the most active geophone (GMax) out of 
# geophones monitoring the longwall;
#  5. gpuls: a number of pulses recorded within previous shift by GMax;
#  6. gdenergy: a deviation of energy recorded within previous shift by GMax from average energy recorded 
# during eight previous shifts;
#  7. gdpuls: a deviation of a number of pulses recorded within previous shift by GMax from average number 
# of pulses recorded during eight previous shifts;
#  8. ghazard: result of shift seismic hazard assessment in the mine working obtained by the 
# seismoacoustic method based on registration coming form GMax only;
#  9. nbumps: the number of seismic bumps recorded within previous shift;
# 10. nbumps2: the number of seismic bumps (in energy range [10^2,10^3)) registered within previous shift;
# 11. nbumps3: the number of seismic bumps (in energy range [10^3,10^4)) registered within previous shift;
# 12. nbumps4: the number of seismic bumps (in energy range [10^4,10^5)) registered within previous shift;
# 13. nbumps5: the number of seismic bumps (in energy range [10^5,10^6)) registered within the last shift;
# 14. nbumps6: the number of seismic bumps (in energy range [10^6,10^7)) registered within previous shift;
# 15. nbumps7: the number of seismic bumps (in energy range [10^7,10^8)) registered within previous shift;
# 16. nbumps89: the number of seismic bumps (in energy range [10^8,10^10)) registered within previous shift;
# 17. energy: total energy of seismic bumps registered within previous shift;
# 18. maxenergy: the maximum energy of the seismic bumps registered within previous shift;
# 19. class: the decision attribute - "1" means that high energy seismic bump occurred in the next shift 
# ("hazardous state"), "0" means that no high energy seismic bumps occurred in the next shift 
# ("non-hazardous state").


## Imports

from __future__ import absolute_import
from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History

import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import h5py
import h52vtp




def getData(datafile):
    '''Function for extracting data from the file and formatting it'''
    with open(datafile, 'rb') as f:
        reader = csv.reader(f)
        data = list(reader)
    
    x,y = [],[]
    for i in range(len(data)):
        row = data[i]
        # convert letters to numbers and reduce the high values
        row[0] = letter2float(row[0])
        row[1] = letter2float(row[1])
        row[2] = letter2float(row[2])
        row[3] = float(row[3])/100000
        row[4] = float(row[4])/1000
        row[5] = float(row[5])/100
        row[6] = float(row[6])/100
        row[7] = letter2float(row[7])
        row[16] = float(row[16])/10000
        row[17] = float(row[17])/10000
        # convert to float
        for j in range(8,16):
            row[j] = float(row[j])+0
        x.append(np.asarray(row[:-1]))
        y.append(row[-1])
    
    return zip(x,y)


def letter2float(letter):
    '''A util for data formatting'''
    if letter == 'a':
        return 0.0
    if letter == 'b':
        return 1.0
    if letter == 'c':
        return 2.0
    if letter == 'd':
        return 3.0
    if letter == 'W':
        return 0.0
    if letter == 'N':
        return 1.0


def make_model(model_type, input_shape, x_train, y_train, x_test, y_test, batch_size, epochs):
    '''This function creates the model and trains it'''
    print(' -- creating model -- ')
    model = Sequential()
    if model_type == 'shallow':
        model.add(Dense(1024,activation='sigmoid',input_shape=input_shape))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
    elif model_type == 'deep':
        model.add(Dense(128,activation='sigmoid',input_shape=input_shape))
        model.add(Dense(64,activation='sigmoid'))
        model.add(Dense(32,activation='sigmoid'))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
    print(' -- training model -- ')
    history = History()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1, # change to 1 to see progress bar during training
              validation_data=(x_test, y_test),
              callbacks=[history])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print(' -- saving model -- ')
    
    # Show accuracy and loss curves
    plt.close('all') # close remaining windows if any
    
    train_accuracy      = history.history['acc']
    train_loss          = history.history['loss']
    validation_accuracy = history.history['val_acc']
    validation_loss     = history.history['val_loss']
    
    plt.figure(0)
    plt.plot(train_accuracy, label='training accuracy')
    plt.plot(validation_accuracy, label='validation accuracy')
    plt.legend(); plt.title('Accuracy'); plt.show()
    plt.figure(1)
    plt.plot(train_loss, label='training loss')
    plt.plot(validation_loss, label='validation loss')
    plt.legend(); plt.title('Loss'); plt.show()
    
    return model


def softmax_loss(x, y):
    '''A util for loss visualization'''
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


def loss_visualisation(model, weight_scale, x_test, y_test):
    '''Function for actually visualising loss once the model is trained'''
    print(" -- starting loss visualization --")
    
    np.random.seed(716)
    # random directions
    delta = weight_scale*np.random.normal(size = np.shape(model))
    eta   = weight_scale*np.random.normal(size = np.shape(model))
    
    x = np.arange(-2,2,0.2)
    y = np.arange(-2,2,0.2)
    X,Y = np.meshgrid(x,y)
    
    weights = model.get_weights()
    
    loss_vals = np.zeros((np.size(x),np.size(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            weights_new = weights + x[i]*delta + y[j]*eta
            model.set_weights(weights_new)
            loss_vals[i,j] = model.evaluate(x_test, y_test, verbose=0)[0]
    
    print(np.shape(loss_vals))
    xcoordinates = np.linspace(-10,10,20)
    ycoordinates = np.linspace(-10,10,20)
    
    f = h5py.File('seismic.h5', 'w')
    f['xcoordinates'] = xcoordinates
    f['ycoordinates'] = ycoordinates
    f['train_loss'] = loss_vals
    f.close()


# get data and shuffle it
data = getData('seismic_bumps.csv')
random.shuffle(data)
# separate training and testing sets
training_data = zip(*data[:int(0.75*len(data))])
testing_data  = zip(*data[int(0.75*len(data)):])
x_train , y_train = np.asarray(training_data[:][0]) , np.asarray(training_data[:][1])
x_test  , y_test  = np.asarray(testing_data[0]) , np.asarray(testing_data[1])
input_shape = np.shape(x_train[0])



# Create and train model
m = make_model('shallow', input_shape, x_train, y_train, x_test, y_test, 64, 300)


weight_scale = 0.1
# Visualize data
loss_visualisation(m, weight_scale, x_test, y_test)




























































