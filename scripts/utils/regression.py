from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from . import metrics as metrics_util
import numpy as np

def get_model(reg_type):
	if reg_type == "linear":
		return LinearRegression()
	elif reg_type == "kneighbor":
		return KNeighborsRegressor()
	elif reg_type == "decisiontree":
		return DecisionTreeRegressor()
	elif reg_type == "randomforest":
		return RandomForestRegressor()
	elif reg_type == "mlp":
		return MLPRegressor()
	else:
		print("unknown regression model type {}".format(reg_type))
		exit(-1)

def train(model, x, y):
	model.fit(x, y)
	return model

def batch_evaluate(model, x, y, metrics):
	print("predicting..")
	y_pred = np.empty(shape=(0, y.shape[1]))
	start = 0
	batch_size = 1024
	while start < len(x):
		print("Evaluating {}".format(start))
		end = (start + batch_size) if (start + batch_size) < len(x) else len(x)
		x_batch = x[start:end]
		output = model.predict(x_batch)
		y_pred = np.append(y_pred, output)
		start = end
	print("getting metrics..")
	metrics_info = metrics_util.compute_metrics(y, y_pred, metrics,
		multioutput="raw_values")
	return (y_pred, metrics_info)

def evaluate(model, x, y, metrics):
	print("predicting..")
	y_pred = model.predict(x)
	print("getting metrics..")
	metrics_info = metrics_util.compute_metrics(y, y_pred, metrics,
		multioutput="raw_values")
	return (y_pred, metrics_info)
