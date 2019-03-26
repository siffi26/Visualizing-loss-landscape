# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 23:29:58 2018

@author: lab716
"""

from builtins import object
import numpy as np
from functions import*

class CPAA_CNN(object):

    def __init__(self, input_dim=(3, 32, 32), NF=32, FS=7,
                 H1=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float64):

        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.grads = {}

        C,H,W = input_dim
        self.params['W1'] = weight_scale*np.random.randn(NF,C,FS,FS)
        self.params['b1'] = np.zeros(NF)
        self.params['W2'] = weight_scale*np.random.randn(NF*H//2*W//2,H1)
        self.params['b2'] = np.zeros(H1)
        self.params['W3'] = weight_scale*np.random.randn(H1,num_classes)
        self.params['b3'] = np.zeros(num_classes)
                 
        self.grads['W1'] = np.zeros((NF,C,FS,FS))
        self.grads['b1'] = np.zeros(NF)
        self.grads['W2'] = np.zeros((NF*H//2*W//2,H1))
        self.grads['b2'] = np.zeros(H1)
        self.grads['W3'] = np.zeros((H1,num_classes))
        self.grads['b3'] = np.zeros(num_classes)
      
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
            
        for k, v in self.grads.items():
            self.grads[k] = v.astype(dtype)


    def loss(self, X, y=None):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        filter_size = W1.shape[2]
#        maintian input spatial dimensions
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        pool_param = {'pool_size': 2}

        pass
        out1,cache1 = conv2D_forward(X,W1,b1,conv_param)
        out2,cache2 = relu_forward(out1)
        out3,cache3 = max_pool_forward(out2,pool_param)
        out4,cache4 = affine_forward(out3,W2,b2)
        out5,cache5 = relu_forward(out4)
        out6,cache6 = affine_forward(out5,W3,b3)
        scores = out6
        
        if y is None:
            return scores

        loss = 0

        loss, dout1 = softmax_loss(scores,y)
        loss = loss + 0.5*self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)) 
        dout2, self.grads['W3'], self.grads['b3'] = affine_backward(dout1,cache6)
        dout3 = relu_backward(dout2,cache5)
        dout4, self.grads['W2'], self.grads['b2'] = affine_backward(dout3,cache4)
        dout5 = max_pool_backward(dout4, cache3)
        dout6 = relu_backward(dout5, cache2)
        dout7, self.grads['W1'], self.grads['b1'] = conv2D_backward(dout6, cache1)

        
        # regularization
        self.grads['W3'] += self.reg*self.params['W3'] 
        self.grads['W2'] += self.reg*self.params['W2'] 
        self.grads['W1'] +=  self.reg*self.params['W1']
        

        return loss, self.grads

class CPAA_CNN_stride(object):

    def __init__(self, input_dim=(3, 32, 32), NF=32, FS=4,
                 H1=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float64):

        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.grads = {}

        C,H,W = input_dim
        self.params['W1'] = weight_scale*np.random.randn(NF,C,FS,FS)
        self.params['b1'] = np.zeros(NF)
        self.params['W2'] = weight_scale*np.random.randn(NF*H//4*W//4,H1)
        self.params['b2'] = np.zeros(H1)
        self.params['W3'] = weight_scale*np.random.randn(H1,num_classes)
        self.params['b3'] = np.zeros(num_classes)
                 
        self.grads['W1'] = np.zeros((NF,C,FS,FS))
        self.grads['b1'] = np.zeros(NF)
        self.grads['W2'] = np.zeros((NF*H//2*W//2,H1))
        self.grads['b2'] = np.zeros(H1)
        self.grads['W3'] = np.zeros((H1,num_classes))
        self.grads['b3'] = np.zeros(num_classes)
      
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
            
        for k, v in self.grads.items():
            self.grads[k] = v.astype(dtype)


    def loss(self, X, y=None):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        filter_size = W1.shape[2]
#        maintian input spatial dimensions
        conv_param = {'stride': 2, 'pad': (filter_size - 1) // 2}

        pool_param = {'pool_size': 2}

        pass
        out1,cache1 = conv2D_forward(X,W1,b1,conv_param)
        out2,cache2 = relu_forward(out1)
        out3,cache3 = max_pool_forward(out2,pool_param)
        out4,cache4 = affine_forward(out3,W2,b2)
        out5,cache5 = relu_forward(out4)
        out6,cache6 = affine_forward(out5,W3,b3)
        scores = out6
        
        if y is None:
            return scores

        loss = 0

        loss, dout1 = softmax_loss(scores,y)
        loss = loss + 0.5*self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)) 
        dout2, self.grads['W3'], self.grads['b3'] = affine_backward(dout1,cache6)
        dout3 = relu_backward(dout2,cache5)
        dout4, self.grads['W2'], self.grads['b2'] = affine_backward(dout3,cache4)
        dout5 = max_pool_backward(dout4, cache3)
        dout6 = relu_backward(dout5, cache2)
        dout7, self.grads['W1'], self.grads['b1'] = conv2D_backward(dout6, cache1)

        
        # regularization
        self.grads['W3'] += self.reg*self.params['W3'] 
        self.grads['W2'] += self.reg*self.params['W2'] 
        self.grads['W1'] +=  self.reg*self.params['W1']
        

        return loss, self.grads
    
class CCPAA_CNN(object):

    def __init__(self, input_dim=(3, 32, 32), NF1 = 32, NF2 = 16, FS1 = 3, FS2 = 3,
                 H1 = 120, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float64):

        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.grads = {}

        C,H,W = input_dim
        self.params['W1'] = weight_scale*np.random.randn(NF1,C,FS1,FS1)
        self.params['b1'] = np.zeros(NF1)
        self.params['W2'] = weight_scale*np.random.randn(NF2,NF1,FS2,FS2)
        self.params['b2'] = np.zeros(NF2)
        self.params['W3'] = weight_scale*np.random.randn(NF2*H//2*W//2,H1)
        self.params['b3'] = np.zeros(H1)
        self.params['W4'] = weight_scale*np.random.randn(H1,num_classes)
        self.params['b4'] = np.zeros(num_classes)
        
        self.grads['W1'] = np.zeros((NF1,C,FS1,FS1))
        self.grads['b1'] = np.zeros(NF1)
        self.grads['W2'] = np.zeros((NF2,NF1,FS2,FS2))
        self.grads['b2'] = np.zeros(NF2)
        self.grads['W3'] = np.zeros((NF2*H//2*W//2,H1))
        self.grads['b3'] = np.zeros(H1)
        self.grads['W4'] = np.zeros((H1,num_classes))
        self.grads['b4'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        
        for k, v in self.grads.items():
            self.grads[k] = v.astype(dtype)


    def loss(self, X, y=None):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        
        FS1 = W1.shape[2]
        FS2 = W2.shape[2]
#        maintian input spatial dimensions
        conv_param1 = {'stride': 1, 'pad': (FS1 - 1) // 2}
        conv_param2 = {'stride': 1, 'pad': (FS2 - 1) // 2}
        pool_param = {'pool_size': 2}

        pass
        out1,cache1 = conv2D_forward(X,W1,b1,conv_param1)
        out2,cache2 = relu_forward(out1)
        out3,cache3 = conv2D_forward(out2,W2,b2,conv_param2)
        out4,cache4 = relu_forward(out3)
        out5,cache5 = max_pool_forward(out4,pool_param)
        out6,cache6 = affine_forward(out5,W3,b3)
        out7,cache7 = relu_forward(out6)
        out8,cache8 = affine_forward(out7,W4,b4)
        scores = out8
        
        if y is None:
            return scores

        loss = 0

        loss, dout1 = softmax_loss(scores,y)
        loss = loss + 0.5*self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)+np.sum(W4**2)) 
        dout2, self.grads['W4'], self.grads['b4'] = affine_backward(dout1,cache8)
        dout3 = relu_backward(dout2,cache7)
        dout4, self.grads['W3'], self.grads['b3'] = affine_backward(dout3,cache6)
        dout5 = max_pool_backward(dout4, cache5)
        dout6 = relu_backward(dout5, cache4)
        dout7, self.grads['W2'], self.grads['b2'] = conv2D_backward(dout6, cache3)
        dout8 = relu_backward(dout7, cache2)
        dout9, self.grads['W1'], self.grads['b1'] = conv2D_backward(dout8, cache1)
        
        # regularization
        self.grads['W4'] += self.reg*self.params['W4'] 
        self.grads['W3'] += self.reg*self.params['W3'] 
        self.grads['W2'] += self.reg*self.params['W2'] 
        self.grads['W1'] +=  self.reg*self.params['W1'] 
        

        return loss, self.grads
    
class CPCPAA_CNN(object):

    def __init__(self, input_dim=(3, 32, 32), NF1 = 32, NF2 = 32, FS1 = 3, FS2 = 3,
                 H1 = 120, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float64):

        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.grads = {}

        C,H,W = input_dim
        self.params['W1'] = weight_scale*np.random.randn(NF1,C,FS1,FS1)
        self.params['b1'] = np.zeros(NF1)
        self.params['W2'] = weight_scale*np.random.randn(NF2,NF1,FS2,FS2)
        self.params['b2'] = np.zeros(NF2)
        self.params['W3'] = weight_scale*np.random.randn(NF2*H//16*W//16,H1)
        self.params['b3'] = np.zeros(H1)
        self.params['W4'] = weight_scale*np.random.randn(H1,num_classes)
        self.params['b4'] = np.zeros(num_classes)
        
        self.grads['W1'] = np.zeros((NF1,C,FS1,FS1))
        self.grads['b1'] = np.zeros(NF1)
        self.grads['W2'] = np.zeros((NF2,NF1,FS2,FS2))
        self.grads['b2'] = np.zeros(NF2)
        self.grads['W3'] = np.zeros((NF2*H//16*W//16,H1))
        self.grads['b3'] = np.zeros(H1)
        self.grads['W4'] = np.zeros((H1,num_classes))
        self.grads['b4'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        
        for k, v in self.grads.items():
            self.grads[k] = v.astype(dtype)


    def loss(self, X, y=None):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        
        FS1 = W1.shape[2]
        FS2 = W2.shape[2]
#        maintian input spatial dimensions
        conv_param1 = {'stride': 2, 'pad': (FS1 - 1) // 2}
        conv_param2 = {'stride': 2, 'pad': (FS2 - 1) // 2}
        pool_param = {'pool_size': 2}

        pass
        out1,cache1 = conv2D_forward(X,W1,b1,conv_param1)
        out2,cache2 = relu_forward(out1)
        out3,cache3 = max_pool_forward(out2,pool_param)
        out4,cache4 = conv2D_forward(out3,W2,b2,conv_param2)
        out5,cache5 = relu_forward(out4)
        out6,cache6 = max_pool_forward(out5,pool_param)
        out7,cache7 = affine_forward(out6,W3,b3)
        out8,cache8 = relu_forward(out7)
        out9,cache9 = affine_forward(out8,W4,b4)
        scores = out9
        
        if y is None:
            return scores

        loss = 0

        loss, dout1 = softmax_loss(scores,y)
        loss = loss + 0.5*self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)+np.sum(W4**2)) 
        dout2, self.grads['W4'], self.grads['b4'] = affine_backward(dout1,cache9)
        dout3 = relu_backward(dout2,cache8)
        dout4, self.grads['W3'], self.grads['b3'] = affine_backward(dout3,cache7)
        dout5 = max_pool_backward(dout4, cache6)
        dout6 = relu_backward(dout5, cache5)
        dout7, self.grads['W2'], self.grads['b2'] = conv2D_backward(dout6, cache4)
        dout8 = max_pool_backward(dout7, cache3)
        dout9 = relu_backward(dout8, cache2)
        dout10, self.grads['W1'], self.grads['b1'] = conv2D_backward(dout9, cache1)
        
        # regularization
        self.grads['W4'] += self.reg*self.params['W4'] 
        self.grads['W3'] += self.reg*self.params['W3'] 
        self.grads['W2'] += self.reg*self.params['W2'] 
        self.grads['W1'] +=  self.reg*self.params['W1'] 
        

        return loss, self.grads
    
class CPCPCPAA_CNN(object):

    def __init__(self, input_dim=(3, 32, 32), NF1 = 16, NF2 = 32, NF3 = 48, FS1 = 3, FS2 = 3, FS3 = 3,
                 H1 = 100, num_classes=10, weight_scale=1e-1, reg=0.0,
                 dtype=np.float64):

        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.grads = {}

        C,H,W = input_dim
        self.params['W1'] = weight_scale*np.random.randn(NF1,C,FS1,FS1)
        self.params['b1'] = np.zeros(NF1)
        self.params['W2'] = weight_scale*np.random.randn(NF2,NF1,FS2,FS2)
        self.params['b2'] = np.zeros(NF2)
        self.params['W3'] = weight_scale*np.random.randn(NF3,NF2,FS3,FS3)
        self.params['b3'] = np.zeros(NF3)
        self.params['W4'] = weight_scale*np.random.randn(NF3*H//8*W//8,H1)
        self.params['b4'] = np.zeros(H1)
        self.params['W5'] = weight_scale*np.random.randn(H1,num_classes)
        self.params['b5'] = np.zeros(num_classes)
        
        self.grads['W1'] = np.zeros((NF1,C,FS1,FS1))
        self.grads['b1'] = np.zeros(NF1)
        self.grads['W2'] = np.zeros((NF2,NF1,FS2,FS2))
        self.grads['b2'] = np.zeros(NF2)
        self.grads['W3'] = np.zeros((NF3,NF2,FS3,FS3))
        self.grads['b3'] = np.zeros(NF3)
        self.grads['W4'] = np.zeros((NF3*H//8*W//8,H1))
        self.grads['b4'] = np.zeros(H1)
        self.grads['W5'] = np.zeros((H1,num_classes))
        self.grads['b5'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        
        for k, v in self.grads.items():
            self.grads[k] = v.astype(dtype)


    def loss(self, X, y=None):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']
        
        FS1 = W1.shape[2]
        FS2 = W2.shape[2]
        FS3 = W3.shape[3]
#        maintian input spatial dimensions
        conv_param1 = {'stride': 1, 'pad': (FS1 - 1) // 2}
        conv_param2 = {'stride': 1, 'pad': (FS2 - 1) // 2}
        conv_param3 = {'stride': 1, 'pad': (FS3 - 1) // 2}
        pool_param = {'pool_size': 2}

        pass
        out1,cache1 = conv2D_forward(X,W1,b1,conv_param1)
        out2,cache2 = relu_forward(out1)
        out3,cache3 = max_pool_forward(out2,pool_param)
        out4,cache4 = conv2D_forward(out3,W2,b2,conv_param2)
        out5,cache5 = relu_forward(out4)
        out6,cache6 = max_pool_forward(out5,pool_param)
        out7,cache7 = conv2D_forward(out6,W3,b3,conv_param3)
        out8,cache8 = relu_forward(out7)
        out9,cache9 = max_pool_forward(out8,pool_param)
        out10,cache10 = affine_forward(out9,W4,b4)
        out11,cache11 = relu_forward(out10)
        out12,cache12 = affine_forward(out11,W5,b5)
        scores = out12
        
        if y is None:
            return scores

        loss = 0

        loss, dout1 = softmax_loss(scores,y)
        loss += 0.5*self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)+np.sum(W4**2)+np.sum(W5**2)) 
        dout2, self.grads['W5'], self.grads['b5'] = affine_backward(dout1,cache12)
        dout3 = relu_backward(dout2,cache11)
        dout4, self.grads['W4'], self.grads['b4'] = affine_backward(dout3,cache10)
        dout5 = max_pool_backward(dout4, cache9)
        dout6 = relu_backward(dout5, cache8)
        dout7, self.grads['W3'], self.grads['b3'] = conv2D_backward(dout6, cache7)
        dout8 = max_pool_backward(dout7, cache6)
        dout9 = relu_backward(dout8, cache5)
        dout10 , self.grads['W2'], self.grads['b2'] = conv2D_backward(dout9, cache4)
        dout11 = max_pool_backward(dout10, cache3)
        dout12 = relu_backward(dout11, cache2)
        dout13, self.grads['W1'], self.grads['b1'] = conv2D_backward(dout12, cache1)
        
        # regularization
        self.grads['W5'] += self.reg*self.params['W5']
        self.grads['W4'] += self.reg*self.params['W4'] 
        self.grads['W3'] += self.reg*self.params['W3'] 
        self.grads['W2'] += self.reg*self.params['W2'] 
        self.grads['W1'] +=  self.reg*self.params['W1'] 
        

        return loss, self.grads
    
class CCPCCPAA_CNN(object):
    """ conv - relu - conv - relu - maxpool - conv - relu -  conv - relu - maxpool -  affine - relu - affine - softmax """
    def __init__(self, input_dim=(3, 32, 32), NF1 = 16, NF2 = 32, NF3 = 32, NF4 = 48, FS1 = 3, FS2 = 3,
                 FS3 = 3, FS4 = 3, H1 = 100, num_classes=10, weight_scale=1e-1, reg=0.0,
                 dtype=np.float64):

        self.params = {}
        self.grads = {}
        self.reg = reg
        self.dtype = dtype

        C,H,W = input_dim
        self.params['W1'] = weight_scale*np.random.randn(NF1,C,FS1,FS1)
        self.params['b1'] = np.zeros(NF1)
        self.params['W2'] = weight_scale*np.random.randn(NF2,NF1,FS2,FS2)
        self.params['b2'] = np.zeros(NF2)
        self.params['W3'] = weight_scale*np.random.randn(NF3,NF2,FS3,FS3)
        self.params['b3'] = np.zeros(NF3)
        self.params['W4'] = weight_scale*np.random.randn(NF4,NF3,FS4,FS4)
        self.params['b4'] = np.zeros(NF4)
        self.params['W5'] = weight_scale*np.random.randn(NF4*H//4*W//4,H1)
        self.params['b5'] = np.zeros(H1)
        self.params['W6'] = weight_scale*np.random.randn(H1,num_classes)
        self.params['b6'] = np.zeros(num_classes)
      
        
        self.grads['W1'] = np.zeros((NF1,C,FS1,FS1))
        self.grads['b1'] = np.zeros(NF1)
        self.grads['W2'] = np.zeros((NF2,NF1,FS2,FS2))
        self.grads['b2'] = np.zeros(NF2)
        self.grads['W3'] = np.zeros((NF3,NF2,FS3,FS3))
        self.grads['b3'] = np.zeros(NF3)
        self.grads['W4'] = np.zeros((NF4,NF3,FS4,FS4))
        self.grads['b4'] = np.zeros(NF4)
        self.grads['W5'] = np.zeros((NF4*H//4*W//4,H1))
        self.grads['b5'] = np.zeros(H1)
        self.grads['W6'] = np.zeros((H1,num_classes))
        self.grads['b6'] = np.zeros(num_classes)
        
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        
        for k, v in self.grads.items():
            self.grads[k] = v.astype(dtype)


    def loss(self, X, y=None):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']
        W6, b6 = self.params['W6'], self.params['b6']
        
        FS1 = W1.shape[2]
        FS2 = W2.shape[2]
        FS3 = W3.shape[2]
        FS4 = W4.shape[2]
#        maintian input spatial dimensions
        conv_param1 = {'stride': 1, 'pad': (FS1 - 1) // 2}
        conv_param2 = {'stride': 1, 'pad': (FS2 - 1) // 2}
        conv_param3 = {'stride': 1, 'pad': (FS3 - 1) // 2}
        conv_param4 = {'stride': 1, 'pad': (FS4 - 1) // 2}
        pool_param = {'pool_size': 2}

        pass
        out1,cache1 = conv2D_forward(X,W1,b1,conv_param1)
        out2,cache2 = relu_forward(out1)
        out3,cache3 = conv2D_forward(out2,W2,b2,conv_param2)
        out4,cache4 = relu_forward(out3)
        out5,cache5 = max_pool_forward(out4,pool_param)
        out6,cache6 = conv2D_forward(out5,W3,b3,conv_param3)
        out7,cache7 = relu_forward(out6)
        out8,cache8 = conv2D_forward(out7,W4,b4,conv_param4)
        out9,cache9 = relu_forward(out8)
        out10,cache10 = max_pool_forward(out9,pool_param)
        out11,cache11 = affine_forward(out10,W5,b5)
        out12,cache12 = relu_forward(out11)
        out13,cache13 = affine_forward(out12,W6,b6)
        scores = out13
        
        if y is None:
            return scores

        loss, grads = 0, {}

        loss, dout1 = softmax_loss(scores,y)
        loss = loss + 0.5*self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)+np.sum(W4**2)+np.sum(W5**2)
                                        +np.sum(W6**2)) 
        dout2, self.grads['W6'], self.grads['b6'] = affine_backward(dout1,cache13)
        dout3 = relu_backward(dout2,cache12)
        dout4, self.grads['W5'], self.grads['b5'] = affine_backward(dout3,cache11)
        dout5 = max_pool_backward(dout4, cache10)
        dout6 = relu_backward(dout5, cache9)
        dout7, self.grads['W4'], self.grads['b4'] = conv2D_backward(dout6, cache8)
        dout8 = relu_backward(dout7, cache7)
        dout9, self.grads['W3'], self.grads['b3'] = conv2D_backward(dout8, cache6)
        dout10 = max_pool_backward(dout9, cache5)
        dout11 = relu_backward(dout10, cache4)
        dout12, self.grads['W2'], self.grads['b2'] = conv2D_backward(dout11, cache3)
        dout13 = relu_backward(dout12, cache2)
        dout14, self.grads['W1'], self.grads['b1'] = conv2D_backward(dout13, cache1)
        

        # regularization
        self.params['W6'] += self.reg*self.params['W6']
        self.params['W5'] += self.reg*self.params['W5']
        self.params['W4'] += self.reg*self.params['W4'] 
        self.params['W3'] += self.reg*self.params['W3'] 
        self.params['W2'] += self.reg*self.params['W2'] 
        self.params['W1'] +=  self.reg*self.params['W1'] 
        

        return loss, self.grads
    
class CPCPCAA_CNN(object):

    def __init__(self, input_dim=(3, 32, 32), NF1 = 16, NF2 = 32, NF3 = 48, FS1 = 3, FS2 = 3, FS3 = 3,
                 H1 = 100, num_classes=10, weight_scale=1e-1, reg=0.0,
                 dtype=np.float64):

        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.grads = {}

        C,H,W = input_dim
        self.params['W1'] = weight_scale*np.random.randn(NF1,C,FS1,FS1)
        self.params['b1'] = np.zeros(NF1)
        self.params['W2'] = weight_scale*np.random.randn(NF2,NF1,FS2,FS2)
        self.params['b2'] = np.zeros(NF2)
        self.params['W3'] = weight_scale*np.random.randn(NF3,NF2,FS3,FS3)
        self.params['b3'] = np.zeros(NF3)
        self.params['W4'] = weight_scale*np.random.randn(NF3*H//4*W//4,H1)
        self.params['b4'] = np.zeros(H1)
        self.params['W5'] = weight_scale*np.random.randn(H1,num_classes)
        self.params['b5'] = np.zeros(num_classes)
        
        self.grads['W1'] = np.zeros((NF1,C,FS1,FS1))
        self.grads['b1'] = np.zeros(NF1)
        self.grads['W2'] = np.zeros((NF2,NF1,FS2,FS2))
        self.grads['b2'] = np.zeros(NF2)
        self.grads['W3'] = np.zeros((NF3,NF2,FS3,FS3))
        self.grads['b3'] = np.zeros(NF3)
        self.grads['W4'] = np.zeros((NF3*H//4*W//4,H1))
        self.grads['b4'] = np.zeros(H1)
        self.grads['W5'] = np.zeros((H1,num_classes))
        self.grads['b5'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        
        for k, v in self.grads.items():
            self.grads[k] = v.astype(dtype)


    def loss(self, X, y=None):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']
        
        FS1 = W1.shape[2]
        FS2 = W2.shape[2]
        FS3 = W3.shape[3]
#        maintian input spatial dimensions
        conv_param1 = {'stride': 1, 'pad': (FS1 - 1) // 2}
        conv_param2 = {'stride': 1, 'pad': (FS2 - 1) // 2}
        conv_param3 = {'stride': 1, 'pad': (FS3 - 1) // 2}
        pool_param = {'pool_size': 2}

        pass
        out1,cache1 = conv2D_forward(X,W1,b1,conv_param1)
        out2,cache2 = relu_forward(out1)
        out3,cache3 = max_pool_forward(out2,pool_param)
        out4,cache4 = conv2D_forward(out3,W2,b2,conv_param2)
        out5,cache5 = relu_forward(out4)
        out6,cache6 = max_pool_forward(out5,pool_param)
        out7,cache7 = conv2D_forward(out6,W3,b3,conv_param3)
        out8,cache8 = relu_forward(out7)
        out9,cache9 = affine_forward(out8,W4,b4)
        out10,cache10 = relu_forward(out9)
        out11,cache11 = affine_forward(out10,W5,b5)
        scores = out11
        
        if y is None:
            return scores

        loss = 0

        loss, dout1 = softmax_loss(scores,y)
        loss += 0.5*self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)+np.sum(W4**2)+np.sum(W5**2)) 
        dout2, self.grads['W5'], self.grads['b5'] = affine_backward(dout1,cache11)
        dout3 = relu_backward(dout2,cache10)
        dout4, self.grads['W4'], self.grads['b4'] = affine_backward(dout3,cache9)
        dout5 = relu_backward(dout4, cache8)
        dout6, self.grads['W3'], self.grads['b3'] = conv2D_backward(dout5, cache7)
        dout7 = max_pool_backward(dout6, cache6)
        dout8 = relu_backward(dout7, cache5)
        dout9 , self.grads['W2'], self.grads['b2'] = conv2D_backward(dout8, cache4)
        dout10 = max_pool_backward(dout9, cache3)
        dout11 = relu_backward(dout10, cache2)
        dout12, self.grads['W1'], self.grads['b1'] = conv2D_backward(dout11, cache1)
        
        # regularization
        self.grads['W5'] += self.reg*self.params['W5']
        self.grads['W4'] += self.reg*self.params['W4'] 
        self.grads['W3'] += self.reg*self.params['W3'] 
        self.grads['W2'] += self.reg*self.params['W2'] 
        self.grads['W1'] +=  self.reg*self.params['W1'] 
        

        return loss, self.grads
    
class CPCPCAA_CNN_stride(object):

    def __init__(self, input_dim=(3, 32, 32), NF1 = 16, NF2 = 32, NF3 = 48, FS1 = 3, FS2 = 3, FS3 = 3,
                 H1 = 100, num_classes=10, weight_scale=1e-1, reg=0.0,
                 dtype=np.float64):

        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.grads = {}

        C,H,W = input_dim
        self.params['W1'] = weight_scale*np.random.randn(NF1,C,FS1,FS1)
        self.params['b1'] = np.zeros(NF1)
        self.params['W2'] = weight_scale*np.random.randn(NF2,NF1,FS2,FS2)
        self.params['b2'] = np.zeros(NF2)
        self.params['W3'] = weight_scale*np.random.randn(NF3,NF2,FS3,FS3)
        self.params['b3'] = np.zeros(NF3)
        self.params['W4'] = weight_scale*np.random.randn(NF3*4,H1)
        self.params['b4'] = np.zeros(H1)
        self.params['W5'] = weight_scale*np.random.randn(H1,num_classes)
        self.params['b5'] = np.zeros(num_classes)
        
        self.grads['W1'] = np.zeros((NF1,C,FS1,FS1))
        self.grads['b1'] = np.zeros(NF1)
        self.grads['W2'] = np.zeros((NF2,NF1,FS2,FS2))
        self.grads['b2'] = np.zeros(NF2)
        self.grads['W3'] = np.zeros((NF3,NF2,FS3,FS3))
        self.grads['b3'] = np.zeros(NF3)
        self.grads['W4'] = np.zeros((NF3*4,H1))
        self.grads['b4'] = np.zeros(H1)
        self.grads['W5'] = np.zeros((H1,num_classes))
        self.grads['b5'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        
        for k, v in self.grads.items():
            self.grads[k] = v.astype(dtype)


    def loss(self, X, y=None):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']
        
        FS1 = W1.shape[2]
        FS2 = W2.shape[2]
        FS3 = W3.shape[3]
#        maintian input spatial dimensions
        conv_param1 = {'stride': 1, 'pad': (FS1 - 1) // 2}
        conv_param2 = {'stride': 2, 'pad': (FS2 - 1) // 2}
        conv_param3 = {'stride': 2, 'pad': (FS3 - 1) // 2}
        pool_param = {'pool_size': 2}

        pass
        out1,cache1 = conv2D_forward(X,W1,b1,conv_param1)
        out2,cache2 = relu_forward(out1)
        out3,cache3 = max_pool_forward(out2,pool_param)
        out4,cache4 = conv2D_forward(out3,W2,b2,conv_param2)
        out5,cache5 = relu_forward(out4)
        out6,cache6 = max_pool_forward(out5,pool_param)
        out7,cache7 = conv2D_forward(out6,W3,b3,conv_param3)
        out8,cache8 = relu_forward(out7)
        out9,cache9 = affine_forward(out8,W4,b4)
        out10,cache10 = relu_forward(out9)
        out11,cache11 = affine_forward(out10,W5,b5)
        scores = out11
        
        if y is None:
            return scores

        loss = 0

        loss, dout1 = softmax_loss(scores,y)
        loss += 0.5*self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)+np.sum(W4**2)+np.sum(W5**2)) 
        dout2, self.grads['W5'], self.grads['b5'] = affine_backward(dout1,cache11)
        dout3 = relu_backward(dout2,cache10)
        dout4, self.grads['W4'], self.grads['b4'] = affine_backward(dout3,cache9)
        dout5 = relu_backward(dout4, cache8)
        dout6, self.grads['W3'], self.grads['b3'] = conv2D_backward(dout5, cache7)
        dout7 = max_pool_backward(dout6, cache6)
        dout8 = relu_backward(dout7, cache5)
        dout9 , self.grads['W2'], self.grads['b2'] = conv2D_backward(dout8, cache4)
        dout10 = max_pool_backward(dout9, cache3)
        dout11 = relu_backward(dout10, cache2)
        dout12, self.grads['W1'], self.grads['b1'] = conv2D_backward(dout11, cache1)
        
        # regularization
        self.grads['W5'] += self.reg*self.params['W5']
        self.grads['W4'] += self.reg*self.params['W4'] 
        self.grads['W3'] += self.reg*self.params['W3'] 
        self.grads['W2'] += self.reg*self.params['W2'] 
        self.grads['W1'] +=  self.reg*self.params['W1'] 
        

        return loss, self.grads
    
class CCPCPAA_CNN(object):
    """ conv - relu - conv - relu - maxpool - conv - relu -  conv - relu - maxpool -  affine - relu - affine - softmax """
    def __init__(self, input_dim=(3, 32, 32), NF1 = 32, NF2 = 32, NF3 = 64, FS1 = 3, FS2 = 3,
                 FS3 = 3, H1 = 300, num_classes=10, weight_scale=1e-1, reg=0.0,
                 dtype=np.float64):

        self.params = {}
        self.grads = {}
        self.reg = reg
        self.dtype = dtype

        C,H,W = input_dim
        self.params['W1'] = weight_scale*np.random.randn(NF1,C,FS1,FS1)
        self.params['b1'] = np.zeros(NF1)
        self.params['W2'] = weight_scale*np.random.randn(NF2,NF1,FS2,FS2)
        self.params['b2'] = np.zeros(NF2)
        self.params['W3'] = weight_scale*np.random.randn(NF3,NF2,FS3,FS3)
        self.params['b3'] = np.zeros(NF3)
        self.params['W4'] = weight_scale*np.random.randn(NF3*H//4*W//4,H1)
        self.params['b4'] = np.zeros(H1)
        self.params['W5'] = weight_scale*np.random.randn(H1,num_classes)
        self.params['b5'] = np.zeros(num_classes)
      
        
        self.grads['W1'] = np.zeros((NF1,C,FS1,FS1))
        self.grads['b1'] = np.zeros(NF1)
        self.grads['W2'] = np.zeros((NF2,NF1,FS2,FS2))
        self.grads['b2'] = np.zeros(NF2)
        self.grads['W3'] = np.zeros((NF3,NF2,FS3,FS3))
        self.grads['b3'] = np.zeros(NF3)
        self.grads['W4'] = np.zeros((NF3*H//4*W//4,H1))
        self.grads['b4'] = np.zeros(H1)
        self.grads['W5'] = np.zeros((H1,num_classes))
        self.grads['b5'] = np.zeros(num_classes)
        
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        
        for k, v in self.grads.items():
            self.grads[k] = v.astype(dtype)


    def loss(self, X, y=None):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']
        
        FS1 = W1.shape[2]
        FS2 = W2.shape[2]
        FS3 = W3.shape[2]
#        maintian input spatial dimensions
        conv_param1 = {'stride': 1, 'pad': (FS1 - 1) // 2}
        conv_param2 = {'stride': 1, 'pad': (FS2 - 1) // 2}
        conv_param3 = {'stride': 1, 'pad': (FS3 - 1) // 2}
        pool_param = {'pool_size': 2}

        pass
        out1,cache1 = conv2D_forward(X,W1,b1,conv_param1)
        out2,cache2 = relu_forward(out1)
        out3,cache3 = conv2D_forward(out2,W2,b2,conv_param2)
        out4,cache4 = relu_forward(out3)
        out5,cache5 = max_pool_forward(out4,pool_param)
        out6,cache6 = conv2D_forward(out5,W3,b3,conv_param3)
        out7,cache7 = relu_forward(out6)
        out8,cache8 = max_pool_forward(out7,pool_param)
        out9,cache9 = affine_forward(out8,W4,b4)
        out10,cache10 = relu_forward(out9)
        out11,cache11 = affine_forward(out10,W5,b5)
        scores = out11
        
        if y is None:
            return scores

        loss, grads = 0, {}

        loss, dout1 = softmax_loss(scores,y)
        loss = loss + 0.5*self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)+np.sum(W4**2)+np.sum(W5**2)) 
        dout2, self.grads['W5'], self.grads['b5'] = affine_backward(dout1,cache11)
        dout3 = relu_backward(dout2,cache10)
        dout4, self.grads['W4'], self.grads['b4'] = affine_backward(dout3,cache9)
        dout5 = max_pool_backward(dout4, cache8)
        dout6 = relu_backward(dout5, cache7)
        dout7, self.grads['W3'], self.grads['b3'] = conv2D_backward(dout6, cache6)
        dout8 = max_pool_backward(dout7, cache5)
        dout9 = relu_backward(dout8, cache4)
        dout10, self.grads['W2'], self.grads['b2'] = conv2D_backward(dout9, cache3)
        dout11 = relu_backward(dout10, cache2)
        dout12, self.grads['W1'], self.grads['b1'] = conv2D_backward(dout11, cache1)
        

        # regularization
        self.params['W5'] += self.reg*self.params['W5']
        self.params['W4'] += self.reg*self.params['W4'] 
        self.params['W3'] += self.reg*self.params['W3'] 
        self.params['W2'] += self.reg*self.params['W2'] 
        self.params['W1'] +=  self.reg*self.params['W1'] 
        

        return loss, self.grads