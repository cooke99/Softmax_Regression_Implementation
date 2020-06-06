# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 20:06:33 2020

@author: cooke99

Standalone Python file so that the SoftmaxRegression class can be 
imported and used in other projects.
"""
import numpy as np

class SoftmaxRegression():
    def __init__(self, penalty='l2', alpha=0.1, eta0 = 0.1):
        if not(penalty in ['l2', None]):
            # This class will only specify an l2 regularization (half the square of the norm of the weights)
            raise Exception('Unrecognized penalty. Only \'l2\' or None penalty is specified for this class.')
        else:
            self.penalty = penalty
            self.alpha = alpha
            self.eta0 = eta0
            print('SoftmaxRegression(penalty={}, alpha={}, eta0={})'.format(penalty, alpha, eta0))
            return None
        
    def fit(self, X, y, iters = 1000):
        # The model will use Batch GD with early stopping to fit the model. Therefore the training
        # data needs to be split into a validation set internally for early stopping.
        # A column of 1s also needs to be appended to the set, which is carried out in function
        # __split_train_val_set:
        X_train,y_train, X_val, y_val = self.__split_train_val_set(X,y)        
        self.K = len(np.unique(y_train)) # No. of classes.
        self.n = np.size(X_train,axis=1) # No. of features.
        self.m = np.size(X_train,axis=0) # No. of training instances.    
        # Randomly initialize k x n array of theta values:
        self.Theta = np.random.randn(self.K, self.n)     
        y_train_prep, y_val_prep = self.__one_hot(y_train), self.__one_hot(y_val) # Encode labels using OneHotEncoding.     
        if self.penalty == None:
            # Rather than 2 large if/else blocks with much the same code (only difference being regularization term),
            # we simply set alpha to 0 if no regularization is desired, which would set the regularization term to 0.
            self.alpha = 0
        # Initialise variables for GD and early stopping:
        min_val_score = 0
        best_theta = np.zeros_like(self.Theta)
        gradients = np.zeros_like(self.Theta)
        epoch_since_min = 0           
        for epoch in range(iters):
            if epoch_since_min >= 100:
                # Breaks the loop if the validation error has increased for the last 100 training epochs.
                break
            feature_weights = np.copy(self.Theta) # Weights vector excluding bias term for regularization.
            feature_weights[:,0] = 0
            p_hat = self.predict(X_train,predict_proba=True, add_bias=False)
            error = p_hat - y_train_prep
            for i in range(self.K):
                gradients[i,:] = error[:,i].reshape(1,-1)@X_train # Gradient sum term
            gradients = gradients*(1/self.m) + self.alpha*feature_weights
            self.Theta = self.Theta - self.eta0*gradients # Gradient descent step              
            y_val_predicts = self.predict(X_val, add_bias=False) # Predictions on validation set.
            score = self.__accuracy(y_val,y_val_predicts) # Compute accuracy on validation set.              
            if score > min_val_score: # If the new theta values give a better accuracy on the validation set.
                best_theta = self.Theta
                min_val_score = score
                epoch_since_min = 0         
            else:
                # Keeps count of the number of training epochs since the last best validation score was found.
                epoch_since_min += 1
        self.Theta = best_theta # The model parameters that gave the best accuracy on the validation set are used.
                  
    def predict(self,X, one_hot = False, predict_proba = False, add_bias = True):
        # The purpose of this function is to make a prediction on an instance X using the parameters Theta. 
        # The optional argument one_hot=False returns an array of labels corresponding to the class index
        # k i.e. for 3 classes it'll return one of [0,1,2]. If one_hot = True, it'll return a OneHot encoded
        # version of the labels.
        # The optional argument predict_proba allows the array of predicted probabilities (p_hat) to be
        # returned from the function.
        # Optional argument add_bias determines whether to concatenate a column of 1s to the X array for 
        # the bias term.
        if one_hot and predict_proba:
            # Both of these options can't be true as it will always return p_hat first and quit
            # the function.
            raise Exception('Can not return one hot encoding of labels and prediction probabilities simultaneously.')
        else:
            pass
        if X.ndim == 1:
            # In case a single instance is passed to the function.
            X = X.reshape(1,-1)
        else:
            pass
        if add_bias:
            X_prep = np.c_[np.ones((np.size(X,axis=0),1)),X]
        else:
            X_prep = X
        softmax_score = X_prep.dot(self.Theta.T)
        exponents = np.exp(softmax_score)
        p_hat = np.divide(exponents, np.sum(exponents, axis=1).reshape(-1,1))        
        if predict_proba:
            return p_hat
        else:    
            predictions = np.argmax(p_hat,axis=1)
            if one_hot:
                return self.__one_hot(predictions.ravel())       
            else:
                return predictions.ravel()
        
    def __split_train_val_set(self, X, y, val_size=0.25):
        # This is a private method that creates a train and validation set from 
        # the data provided using a 75/25 split. X and y are assumed to be numpy arrays.
        X_prep = np.c_[np.ones((np.size(X,axis=0),1)),X] # Concatenate bias term to data.
        shuffled_indices = np.random.permutation(len(y))
        split_index = np.int_(val_size*len(y))
        train_indices = shuffled_indices[split_index:]
        val_indices = shuffled_indices[:split_index]   
        return X_prep[train_indices],y[train_indices], X_prep[val_indices], y[val_indices]
    
    def __one_hot(self,labels):
        # This private method is a custom implementation for OneHotEncoder that will convert the labels array
        # of k values (class indexes) into a one hot encoded matrix.
        encoded = np.zeros((len(labels),self.K))
        encoded[np.arange(len(labels)), labels] = 1
        return encoded
    
    def __accuracy(self,labels,predictions):
        # Computes the accuracy to evaluate generalization score on validation set.
        if labels.shape != predictions.shape:
            raise Exception('Error: Label and prediction dimensions do not match')
        else:
            correct = np.sum(labels==predictions,axis=0)
            return correct/(np.size(labels,axis=0))