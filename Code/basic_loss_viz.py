# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 19:19:14 2019

@author: Lab716A-PC
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:09:27 2019

@author: User
"""
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from functions import softmax_loss
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from cs231n.data_utils import get_CIFAR10_data


#mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
#
#num_train = mnist.train.images.shape[0]
#num_val = mnist.validation.images.shape[0]
#num_test =  mnist.test.images.shape[0]
#
#data = {
#'X_train' : mnist.train.images[:num_train][:,None,:].reshape(num_train,1,28,28),
#'y_train' : mnist.train.labels[:num_train],
#'X_val' : mnist.validation.images[:num_val][:,None,:].reshape(num_val,1,28,28),
#'y_val' : mnist.validation.labels[:num_val],
#'X_test' : mnist.test.images[:num_test][:,None,:].reshape(num_test,1,28,28),
#'y_test' : mnist.test.labels[:num_test]
#}

CIFARdata = get_CIFAR10_data()

num_train = data['X_train'].shape[0]
#num_train = N
data = {
  'X_train': CIFARdata['X_train'][:num_train],
  'y_train': CIFARdata['y_train'][:num_train],
  'X_val': CIFARdata['X_val'],
  'y_val': CIFARdata['y_val'],
  'X_test': CIFARdata['X_test'],
  'y_test': CIFARdata['y_test'],
}
# scale of random directions
weight_scale = 1e-1
w3_time = []
model_time = []
# load models and save weights from different models
for i in range(2):
    model_file = np.load('CIFARmodel1_iter_%d.npz'%(i*100))
    model_time.append(model_file['arr_0'][()])
    w3_time.append(model_time[-1].params['W3'])
    
# model selector
t = 0

w3 = w3_time[t]

np.random.seed(716)
# random directions
w3_dir1 = weight_scale*np.random.normal(size = np.shape(w3))
w3_dir2 = weight_scale*np.random.normal(size = np.shape(w3))

N = 200 # no of points in the grid
x = np.arange(-2,2,0.2)
y = np.arange(-2,2,0.2)
X,Y = np.meshgrid(x,y)

loss_vals = np.zeros((np.size(x),np.size(y)))
for i in range(len(x)):
    for j in range(len(y)):
        w3_new = w3 + x[i]*w3_dir1 + y[j]*w3_dir2
        model_time[t].params['W3'] = w3_new
        scores = model_time[t].loss(data['X_val'][:N])
        loss_vals[i][j],_ = softmax_loss(scores, data['y_val'][:N])
        print('Point %d/%d'%(i*len(x)+j,len(x)*len(y)))

# surface plot     
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,loss_vals)
#contour plot
vmin=np.min(loss_vals)
vmax=np.max(loss_vals)
vlevel= (np.max(loss_vals) - np.min(loss_vals))/20
fig = plt.figure()
CS = plt.contour(X,Y,loss_vals,cmap='summer', levels=np.arange(vmin, vmax, vlevel))
plt.clabel(CS, inline=1, fontsize=8)




