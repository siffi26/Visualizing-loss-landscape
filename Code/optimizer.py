# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 23:21:21 2018

@author: User
"""
import numpy as np

class optimizer(object):
    
    def __init__(self, model, data, **kwargs):
        
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.lr = kwargs.pop('learning_rate', 1e-3)
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
#        self.num_train_samples = kwargs.pop('num_train_samples', 1000)
        self.verbose = kwargs.pop('verbose', True)
        self.num_train = self.X_train.shape[0]
        self.iterations_per_epoch = max(self.num_train // self.batch_size, 1)
        self.num_iterations = self.num_epochs * self.iterations_per_epoch
        self.train_acc_history = []
        self.val_acc_history = []
        self.loss_history = []
        self.best_val_acc = 0
        self.batch_grads = []
        self.best_params = {}
        self.test = kwargs.pop('test_set', False)
        self.save_model = kwargs.pop('save_model', False)
        
        if self.save_model:
            self.model_name = kwargs.pop('model_name')

        if self.test:
            self.X_test = data['X_test']
            self.y_test = data['y_test']
            self.test_acc_history = []
            

    def iterate(self):
    
        # Make a minibatch of training data
        batch_mask = np.random.choice(self.num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]
        
        # Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)
        
        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            self.model.params[p]= w - self.lr*dw

    def check_accuracy(self, X, y, batch_size=100):
    
            N = X.shape[0]
            # Compute predictions in batches
            num_batches = N // batch_size
            if N % batch_size != 0:
                num_batches += 1
            y_pred = []
            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                scores = self.model.loss(X[start:end])
                y_pred.append(np.argmax(scores, axis=1))    
            y_pred = np.hstack(y_pred)
            acc = np.mean(y_pred == y)
            return acc
    
    def train(self):
        self.epoch = 0
        for t in range(self.num_iterations):
            self.iterate()
            epoch_end = (t + 1) % self.iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                self.lr = self.lr*self.lr_decay 
                
            if self.save_model:
                if t%50 == 0:
    #                np.savez('model_params_iter_%d.npz'% t, self.model.params)
    #                np.savez('CIFARmodel1_iter_%d.npz'% t, self.model)
                    np.savez('%s_iter_%d.npz'% (self.model_name, t), self.model)
                
            first_it = (t == 0)
            last_it = (t == self.num_iterations - 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train)                                      
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                if self.test:
                    test_acc = self.check_accuracy(self.X_test, self.y_test)
                    self.test_acc_history.append(test_acc)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                if self.verbose and self.test:
                    print('(Epoch %d / %d) train acc: %f; val_acc: %f; test_acc: %f' % (self.epoch,self.num_epochs, train_acc, val_acc, test_acc))
                elif self.verbose:  
                    print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (self.epoch,self.num_epochs, train_acc, val_acc))
                
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()
            