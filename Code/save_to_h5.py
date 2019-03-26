# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 14:27:47 2019

@author: User
"""
import numpy as np
import h5py
import h52vtp

lossfile = np.load('MNIST_FF4_2W_10_11_Random_Loss.npz')
loss = lossfile['arr_0']
xcoordinates = np.linspace(-10,10,100)
ycoordinates = np.linspace(-10,10,100)

surf_file = 'surf_MNIST_FF4_2W_10_11_Random.h5'
f = h5py.File(surf_file, 'w')
f['xcoordinates'] = xcoordinates
f['ycoordinates'] = ycoordinates
f['train_loss'] = loss
f.close()


