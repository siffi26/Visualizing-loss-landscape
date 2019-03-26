# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:27:56 2018

@author: User
"""
import numpy as np
from numba import jit
from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
import platform

@jit(nopython=True,cache = True)
def im2col_numba(Xpad,Xcol,N,HO,WO,C,FH,FW,S):

    for i0 in range(N):
        for i1 in range(HO):
            r = i1*S
            for i2 in range(WO):
                c = i2*S
                for i3 in range(C):
                    for i4 in range(0,FH):
                        for i5 in range(0,FW):
                            Xcol[i3*FH*FW + i4*FW + i5 , i0*HO*WO+ i1*WO + i2] = Xpad[i0,i3,r+i4,c+i5]
    
    return Xcol

@jit(nopython=True,cache = True)
def col2im_numba(dXcol, dXpad, N,C,HH,WW,FH,FW,P,S):
    
    HO = (HH-FH+2*P)//S+1
    WO = (WW-FW+2*P)//S+1    
    for i0 in range(C):
        for i1 in range(FH):
            for i2 in range(FW):
                r = i0*FW*FH + i1*FW + i2
                for i3 in range(HO):
                    for i4 in range(WO):
                        for i5 in range(N):
                            c = i3*WO*N + i4*N + i5 
                            dXpad[i5,i0,S*i3+i1,S*i4+i2] += dXcol[r,c]
    
    if P > 0:
        dX = dXpad[:,:,P:-P,P:-P]

    return dX

def conv2D_forward(X,W,b,conv_param):
    N,C,HH,WW = np.shape(X)
    F,_,FH,FW = np.shape(W)
    S = conv_param['stride']
    P = conv_param['pad']
    assert (HH-FH+2*P)%S == 0 ,'Invalid filter height'
    assert (WW-FW+2*P)%S == 0 ,'Invalid filter width'
    HO = (HH-FH+2*P)//S+1
    WO = (WW-FW+2*P)//S+1
    Xpad = np.pad(X,((0,0),(0,0),(P,P),(P,P)),'constant',constant_values=0)
    Xcol = np.zeros((FH*FW*C,HO*WO*N))
    Xcol = im2col_numba(Xpad,Xcol,N,HO,WO,C,FH,FW,S)                                
    conv = W.reshape(F,-1)@Xcol + b.reshape(-1,1)
    conv.shape = (F,N,HO,WO)
    out = conv.transpose(1,0,2,3)
    out = np.ascontiguousarray(out,dtype=X.dtype)
    cache = (X,W,b,conv_param,Xcol)
    return out,cache
  
def conv2D_backward(dout,cache):
    X, W, b, conv_param, Xcol = cache
    N,C,HH,WW = np.shape(X)
    F,_,FH,FW = np.shape(W)
    S = conv_param['stride']
    P = conv_param['pad']
    db = np.sum(dout, axis=(0, 2, 3))
    dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
    dW = dout_reshaped.dot(Xcol.T).reshape(W.shape)
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(F, -1)
    dXcol = W.reshape(F, -1).T.dot(dout_reshaped)
    if P > 0:
        dXpad = np.zeros((N,C,HH+2*P,WW+2*P))
    else:
        dXpad = np.zeros_like(X)   
    dX = col2im_numba(dXcol, dXpad, N,C,HH,WW,FH,FW,P,S)
    return dX, dW, db


def max_pool_forward(X, pool_param):
    N,C,HH,WW = np.shape(X)
    FH = pool_param['pool_size']
    FW = FH
    assert HH % FH == 0 ,'Invalid pooling filter size'
    HO = HH //FH
    WO = WW //FW
    Xpatch = X.reshape(N,C,HO,FH,WO,FW)
    out = Xpatch.max(axis=3).max(axis=4)
    cache = (X,Xpatch,out)
    return out,cache

def max_pool_backward(dout,cache):
    X, Xpatch, out = cache
    dXpatch = np.zeros_like(Xpatch)
    out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
    mask = (Xpatch == out_newaxis)
    dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
    dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dXpatch)
    dXpatch[mask] = dout_broadcast[mask]
    dXpatch /= np.sum(mask, axis=(3, 5), keepdims=True)
    dX = dXpatch.reshape(X.shape)
    return dX

def relu_forward(X):
    out = np.zeros_like(X)
    mask = np.where(X > 0)
    out[mask] = X[mask]
    cache = X,out
    return out,cache

def relu_backward(dout,cache):
    X,out = cache
    dX = np.zeros_like(X)
    mask = out == X
    dX[mask] = 1
    dX = dX*dout
    return dX

def affine_forward(X,W,b):
    N = X.shape[0]
    M = W.shape[1]
    out = X.reshape(N,-1).dot(W)+b.reshape(-1,M)
    cache = (X,W,b)
    return out,cache

def affine_backward(dout,cache):
    X,W,b = cache
    N = X.shape[0]
    X_reshaped = X.reshape(N,-1)
    dX = dout.dot(W.T).reshape(X.shape)
    dW = X_reshaped.T.dot(dout)
    db = np.sum(dout,axis = 0)
    return dX,dW,db 

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax_loss(x, y):
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

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    
    # Load the raw CIFAR-10 data
    cifar10_dir = 'datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }

