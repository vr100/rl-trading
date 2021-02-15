import utils.regression as regression
import utils.misc as helper
import utils.metrics as metrics_util
from sklearn.ensemble import VotingRegressor, AdaBoostRegressor, StackingRegressor, BaggingRegressor
import numpy as np

def _get_estimators(x_size, y_size, config):
	estimator_count = config["estimator_count"]
	estimators = []
	for i in range(estimator_count):
		model = regression.get_model(x_size, y_size, config)
		estimators.append((f"{i}", model))
	return estimators

def get_model(x_size, y_size, config):
	ensemble_type = config["ensemble_type"]
	if ensemble_type == "voting":
		estimators = _get_estimators(x_size, y_size, config)
		return VotingRegressor(estimators)
	elif ensemble_type == "adaboost":
		base_estimator = regression.get_model(x_size, y_size, config)
		return AdaBoostRegressor(base_estimator=base_estimator,
			n_estimators=config["estimator_count"], random_state=0)
	elif ensemble_type == "stacking":
		estimators = _get_estimators(x_size, y_size, config)
		return StackingRegressor(estimators=estimators, cv=10,
			n_jobs=config["job_count"], passthrough=True,
			verbose=1)
	elif ensemble_type == "bagging":
		base_estimator = regression.get_model(x_size, y_size, config)
		return BaggingRegressor(base_estimator=base_estimator,
			n_estimators=config["estimator_count"],
			n_jobs=config["job_count"], random_state=0,
			verbose=1)
	else:
		print("Unknown ensemble {}".format(ensemble_type))
		exit()

def train(model, x, y):
	if not isinstance(x, np.ndarray):
		x = x.to_numpy()
		y = y.to_numpy()
	x = helper.add_gaussian_noise(x)
	model.fit(x, y)
	return model

def evaluate(model, x, y, metrics):
	y_pred = model.predict(x)
	metrics_info = metrics_util.compute_metrics(y, y_pred, metrics,
		multioutput="raw_values")
	return (y_pred, metrics_info)

def infer(model, x, config):
	y_pred = model.predict(x)
	return y_pred
