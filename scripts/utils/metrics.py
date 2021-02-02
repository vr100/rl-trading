from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score

def compute_metrics(actual, pred, metric_names, multioutput="uniform_average"):
	metrics = {}
	for m in metric_names:
		if m == "mse":
			mse = mean_squared_error(actual, pred, multioutput=multioutput,
				squared=True)
			metrics[m] = mse.tolist()
		elif m == "rmse":
			rmse = mean_squared_error(actual, pred, multioutput=multioutput,
				squared=False)
			metrics[m] = rmse.tolist()
		elif m == "r2":
			r2 = r2_score(actual, pred, multioutput=multioutput)
			metrics[m] = r2.tolist()
		elif m == "mae":
			mae = mean_absolute_error(actual, pred, multioutput=multioutput)
			metrics[m] = mae.tolist()
		elif m == "mape":
			mape = mean_absolute_percentage_error(actual, pred,
				multioutput=multioutput)
			metrics[m] = mape.tolist()
		elif m == "ev":
			ev = explained_variance_score(actual, pred, multioutput=multioutput)
			metrics[m] = ev.tolist()
	return metrics
