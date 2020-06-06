# Softmax_Regression_Implementation
This repo contains a custom implementation of SoftmaxRegression with Batch Gradient Descent and early stopping using just NumPy.

## Files in this repo:

`Softmax_reg_implementation.ipynb` - Notebook detailing the implementation and analysis of the model, including a comparison with Scikit-learn's implementation of Softmax Regression.

`SoftmaxRegression.py` - Python file containing only the class so that it can be imported and used.

## Using the SoftmaxRegression model:

The SoftmaxRegression class can be initialised with 3 arguments (the default values are shown in all examples below):

`soft_reg = SoftmaxRegression(penalty='l2', alpha=0.1, eta0 = 0.1)`

- `penalty` specifies the type of regularization to use. Only 2 options are available - `'l2'` and `None`. These options correspond to those available in Scikit-learn.

- `alpha` specifies the regularization parameter.

- `eta0` specifies the step size to be used during Batch Gradient Descent when training the model.

The SoftmaxRegression class has 2 methods available to the user: `fit` and `predict`:

`soft_reg.fit(X, y, iters = 1000)`

- X and y are the training data provided to the model. 
- `iters` specifies the maximum number of iterations to run Gradient Descent.

`soft_reg.predict(X, y, one_hot = False, predict_proba = False, add_bias = True)`

- X is the test data to make predictions on.
- `one_hot` is a flag that determines whether the predictions are OneHotEncoded or not. If set to False, the model returns the predicted class index (e.g. for 3 classes it would return one of k = [0,1,2]).
- `predict_proba` is a flag that determines whether the prediction probabilities p_hat are returned. If set to False, the same applies as above. This flag can be used to return the prediction probabilities to compute the cost J.
- `add_bias` determines whether a column of 1s needs to be concatenated to X before the prediction is made (this is necessary to compute the Softmax Score).

Don't touch the parameters if you don't know what they do or you're gonna have a bad time man.
