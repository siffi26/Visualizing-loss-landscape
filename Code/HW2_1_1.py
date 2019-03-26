# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 20:18:53 2018

@author: User
"""
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import CNNs
import matplotlib.pyplot as plt
import optimizer
from time import time

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

num_train = mnist.train.images.shape[0]
num_val = mnist.validation.images.shape[0]
num_test =  mnist.test.images.shape[0]

dataMNIST = {
'X_train' : mnist.train.images[:num_train][:,None,:].reshape(num_train,1,28,28),
'y_train' : mnist.train.labels[:num_train],
'X_val' : mnist.validation.images[:num_val][:,None,:].reshape(num_val,1,28,28),
'y_val' : mnist.validation.labels[:num_val],
'X_test' : mnist.test.images[:num_test][:,None,:].reshape(num_test,1,28,28),
'y_test' : mnist.test.labels[:num_test]
}

for k, v in dataMNIST.items():
  print('%s: ' % k, v.shape)

  
#modelMNIST = MNIST_CNN(weight_scale=1e-1, reg = 0.00)
#MNISTmodel1 = CNNs.CPCPAA_CNN(input_dim = (1,28,28), weight_scale=1e-2, NF1 = 32, NF2 = 32, FS1 = 3, FS2 = 3,
#                 H1 = 500, reg = 0.00)
  
#model1 = CNNs.CPAA_CNN(input_dim = (1,28,28), weight_scale=1e-1, NF = 32, FS = 3, H1 = 100)
model3 = CNNs.CPCPCAA_CNN(input_dim = (1,28,28), weight_scale=1e-1, reg = 0.00, FS1 = 5, FS2 = 5,  FS3 = 5, 
                     NF1 = 16, NF2 = 32, NF3 = 32, H1 = 100, dtype=np.float64)

opt1 = optimizer.optimizer(model3 , dataMNIST, num_epochs = 30, learning_rate = 3e-3, lr_decay = 0.97,
                      batch_size= 100, test_set = True, save_model = True , model_name = 'MNISTmodel4')

t0 = time()
opt1.train()
t1 = time()

#plt.figure()            
##plt.subplot(2, 1, 1)
#plt.plot(opt1.loss_history, )
#plt.xlabel('iteration')
#plt.ylabel('loss')
#plt.title('Learning curve')
#plt.savefig('MNIST4x4_Strided_Loss.png',dpi=250,bbox_inches = 'tight')
#
#plt.figure()  
##plt.subplot(2, 1, 2)
#plt.plot(opt1.train_acc_history, '-o')
#plt.plot(opt1.val_acc_history, '-o')
#plt.plot(opt1.test_acc_history, '-o')
#plt.legend(['train', 'val', 'test'], loc='upper left')
#plt.xlabel('epoch')
#plt.ylabel('accuracy')
#plt.title('Performance')
#plt.savefig('MNIST4x4_Strided_Acc.png',dpi=250,bbox_inches = 'tight')
#plt.show()

#loss1 = opt1.loss_history
#train_acc1 = opt1.train_acc_history
#val_acc1 = opt1.val_acc_history
#
#W1 = opt1.best_params['W1']
#W2 = opt1.best_params['W2']
#W3 = opt1.best_params['W3']
##W4 = opt1.best_params['W4']
#b1 = opt1.best_params['b1']
#b2 = opt1.best_params['b2']
#b3 = opt1.best_params['b3']
#b4 = opt1.best_params['b4']

#plt.figure()
#histW1, bins = np.histogram(W1.reshape(1,-1),bins = 20)
#plt.bar(bins[:-1],histW1)
#plt.xlabel('Value')
#plt.ylabel('Number')
#plt.title('Convolutional  Layer 1')
#plt.savefig('MNIST4x4_Strided_HistW1.png',dpi=250,bbox_inches = 'tight')
#
#plt.figure()
#histW2, bins = np.histogram(W2.reshape(1,-1),bins = 20)
#plt.bar(bins[:-1],histW2)
#plt.xlabel('Value')
#plt.ylabel('Number')
#plt.title('Convolutional Layer 2')
#plt.savefig('MNIST4x4_Strided_HistW2.png',dpi=250,bbox_inches = 'tight')
#
#plt.figure()
#histW3, bins = np.histogram(W3.reshape(1,-1),bins = 20)
#plt.bar(bins[:-1],histW3)
#plt.xlabel('Value')
#plt.ylabel('Number')
#plt.title('Dense Layer 1')
#plt.savefig('MNIST4x4_Strided_HistW3.png',dpi=250,bbox_inches = 'tight')
#
#plt.figure()
#histW4, bins = np.histogram(W4.reshape(1,-1),bins = 20)
#plt.bar(bins[:-1],histW4)
#plt.xlabel('Value')
#plt.ylabel('Number')
#plt.title('Dense Layer 2')
#plt.savefig('MNIST4x4_Strided_HistW4.png',dpi=250,bbox_inches = 'tight')
##display(modelMNIST.params['W1'])
#display(modelMNIST.params['W2'])