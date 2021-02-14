from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import utils.metrics as metrics_util
import utils.mlp as mlp
import utils.misc as helper
import numpy as np

def get_model(reg_type, x_size, y_size, job_count):
	if reg_type == "linear":
		return LinearRegression()
	elif reg_type == "ridge":
		return Ridge()
	elif reg_type == "lasso":
		return Lasso()
	elif reg_type == "elastic":
		return ElasticNet()
	elif reg_type == "kneighbor":
		return KNeighborsRegressor()
	elif reg_type == "decisiontree":
		return DecisionTreeRegressor()
	elif reg_type == "randomforest":
		return RandomForestRegressor(n_jobs=job_count,
			max_depth=20, random_state=0)
	elif reg_type == "mlp":
		return mlp.get_model(x_size, y_size)
	else:
		print("unknown regression model type {}".format(reg_type))
		exit(-1)

def train(model, x, y):
	if isinstance(model, mlp.MLP):
		return mlp.train(model, x, y)
	if not isinstance(x, np.ndarray):
		x = x.to_numpy()
		y = y.to_numpy()
	x = helper.add_gaussian_noise(x)
	model.fit(x, y)
	return model

def evaluate(model, x, y, metrics):
	if isinstance(model, mlp.MLP):
		return mlp.evaluate(model, x, y, metrics)
	y_pred = model.predict(x)
	metrics_info = metrics_util.compute_metrics(y, y_pred, metrics,
		multioutput="raw_values")
	return (y_pred, metrics_info)

def infer(model, x, output_size=1):
	if isinstance(model, mlp.MLP):
		return mlp.infer(model, x, output_size)
	y_pred = model.predict(x)
	return y_pred
