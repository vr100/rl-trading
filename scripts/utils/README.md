# Utility scripts

## autoencoder.py

This script supports EnhancingAutoEncoder and ReducingAutoEncoder

	EnhancingAutoEncoder:
		internal hidden layer size > input size
	ReducingAutoEncoder:
		internal hidden layer size < input size

## dataset.py

This script supports three functionalities:

* splitting given dataset into train and test, storing the two datasets as csv in given folder
* reading given train and test csv files into memory, filling in nan values with mean / zero
* reading given train and test csv files into memory, choosing specified amount of random data from the same

## ensemble_regression.py

This script supports ensemble methods on regression algorithms. Supported ensemble methods are

* voting
* adaboost
* stacking
* bagging

## hyperparams.py

This is a utility script for hyperparam optimization. This supports

* loading of hyperparam config file
* applying hyperparam optimization function with a given training function and dataset

## lstm.py

This script support LSTM model

## metrics.py

This script enables computation of metrics between predicted and observed values. Supported metrics are

* Mean Square Error
* Root Mean Square Error
* R2 score
* Mean Absolute Error
* Mean Absolute Percentage Error
* Explained Variance Score

## misc.py

This is a utility script. It supports

* adding gaussian noise to given data
* converting given data to torch representation

## mlp.py

This is a custom implementation of MLP NN model

## regression.py

This script supports regression algorithms. Supported algorithms are:

* Linear
* Ridge
* Lasso
* ElasticNet
* K Neighbor
* Decision Tree
* Random Forest
* MLP

## residuals.py

This script supports analysis of residuals using plot. The supported plots are

* scatter plot between predicted value and residuals
* normalize plot for residuals
* quantile plot for residuals

## rl.py

This scripts supports reinforcement learning algorithms. Supported algorithms are:

* A2C
* PPO
* DDPG
* SAC

## scaler.py

This script supports processing of data using different transformations. Supported scaling algorithms are:

* robust
* standard
* quantile
* power
* identity
* minmax
