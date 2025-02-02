import argparse, os, json, joblib, torch, time
from utils import regression, dataset, autoencoder, scaler, ensemble_regression, residuals

X_SKIP_COLS = ["date", "weight", "ts_id", "resp", "resp_1", "resp_2", "resp_3", "resp_4"]
Y_OUTPUT_COLS = ["date", "ts_id"]
METRICS_INFO = ["mse", "rmse", "r2", "ev", "mae", "mape"]

def load_model(model_path):
	model = torch.load(model_path)
	model.eval()
	return model

def prepare_data(data_folder, model_path, config, fast_mode):
	y_cols = config["y_cols"]
	(train, test, na_value) = dataset.read_data(data_folder,
		fast_mode, na_value=config["na_value"])
	if "selected_features" in config:
		x_train = train[config["selected_features"]]
		x_test = test[config["selected_features"]]
	else:
		x_train = train.drop(X_SKIP_COLS, axis=1)
		x_test = test.drop(X_SKIP_COLS, axis=1)
	y_train = train[y_cols]
	y_test = test[y_cols]
	out_train = train[Y_OUTPUT_COLS]
	out_test = test[Y_OUTPUT_COLS]

	x_scaler_type = config["x_scaler_type"]
	(x_train, x_scaler) = scaler.scale_data(x_train,
		scaler_type=x_scaler_type)
	(y_train, y_scaler) = scaler.scale_data(y_train)
	(x_test, x_scaler) = scaler.scale_data(x_test, x_scaler)
	(y_test, y_scaler) = scaler.scale_data(y_test, y_scaler)

	if model_path is not None:
		print("Encoding data...")
		model = load_model(model_path)
		x_train = autoencoder.encode(model, x_train,
			config["autoencoder_output_features"])
		x_test = autoencoder.encode(model, x_test,
			config["autoencoder_output_features"])

	train_data = { "x": x_train, "y": y_train, "out": out_train }
	test_data = { "x": x_test, "y": y_test, "out": out_test }
	data_scaler = { "x": x_scaler, "y": y_scaler }

	return (train_data, test_data, data_scaler, na_value)

def postprocess_data(out_data, y_pred, config):
	y_cols = config["y_cols"]
	y_out = out_data.copy()
	if len(y_cols) == 1:
		col = y_cols[0]
		y_out[col] = y_pred
	else:
		index = 0
		for col in y_cols:
			y_out[col] = y_pred[:,index]
			index = index + 1
	return y_out

def regression_path(train, test, config, x_size, y_size):
	model = regression.get_model(x_size, y_size, config)
	print("Training...")
	model = regression.train(model, train["x"], train["y"], config)
	print("Evaluating...")
	(y_pred, metrics) = regression.evaluate(model, test["x"],
		test["y"], METRICS_INFO, config)
	return (model, y_pred, metrics)

def ensemble_path(train, test, config, x_size, y_size):
	model = ensemble_regression.get_model(x_size, y_size, config)
	print("Training...")
	model = ensemble_regression.train(model, train["x"], train["y"])
	print("Evaluating...")
	(y_pred, metrics) = ensemble_regression.evaluate(model, test["x"],
		test["y"], METRICS_INFO)
	return (model, y_pred, metrics)

def train_evaluate(data_folder, output_folder, autoencoder_path,
	config, fast_mode):
	print("Preparing data...")
	(train, test, scalers, na_value) = prepare_data(data_folder,
		autoencoder_path, config, fast_mode)
	y_size = len(config["y_cols"])
	x_size = train["x"].shape[1] if autoencoder_path is None else \
		config["autoencoder_output_features"]
	if config["ensemble_type"] == "none":
		(model, y_pred, metrics) = regression_path(train, test,
			config, x_size, y_size)
	else:
		(model, y_pred, metrics) = ensemble_path(train, test,
			config, x_size, y_size)

	print("Postprocessing data...")

	if not os.path.exists(output_folder):
		print(f"folder: {output_folder} does not exist, creating...")
		os.mkdir(output_folder)

	y_output = postprocess_data(test["out"], y_pred, config)
	output_path = os.path.join(output_folder, "pred.csv")
	y_output.to_csv(output_path, index=False)

	# plotting residual related analysis
	residuals.plot(test["y"], y_pred, output_folder)

	result = { "metrics": metrics, "input_config": config,
		"fast_mode": fast_mode, "na_value": na_value }
	result_path = os.path.join(output_folder, "result.json")
	json_config = json.dumps(result, indent=4)
	with open(result_path, "w") as result_file:
		result_file.write(json_config)

	model_path = os.path.join(output_folder, "regression-model.joblib")
	joblib.dump(model, model_path)

	x_scaler_path = os.path.join(output_folder, "x-scaler.joblib")
	joblib.dump(scalers["x"], x_scaler_path)
	y_scaler_path = os.path.join(output_folder, "y-scaler.joblib")
	joblib.dump(scalers["y"], y_scaler_path)

	print("Output files (model, result, prediction) saved to {}".format(
		output_folder))

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--data_path", type=str, help="specifies the data folder path",
		required=True)
	parser.add_argument(
		"--output_path", type=str, help="specifies the output folder path",
		required=True)
	parser.add_argument(
		"--config_path", type=str, help="specifies the json config path",
		required=True)
	parser.add_argument(
		"--features_path", type=str, help="specifies the boruta algo feature selection results json file path",
		default=None)
	parser.add_argument(
		"--autoencoder_path", type=str, help="specifies the autoencoder model path",
		required=False, default=None)
	parser.add_argument(
		"--fast_mode", default=False,
		type=lambda s: s.lower() in ['true', 'yes', '1'],
		help="specifies whether only a sample subset should be run")
	return vars(parser.parse_args())

def main():
	args = parse_args()
	print("Args: {}".format(args))
	data_path = os.path.abspath(args["data_path"])
	output_path = os.path.abspath(args["output_path"])
	config_path = os.path.abspath(args["config_path"])
	autoencoder_path = None if args["autoencoder_path"] is None else \
		os.path.abspath(args["autoencoder_path"])
	fast_mode = args["fast_mode"]
	with open(config_path, "r") as json_file:
		config = json.load(json_file)
	if args["features_path"] is not None:
		features_path = os.path.abspath(args["features_path"])
		with open(features_path, "r") as json_file:
			features = json.load(json_file)
		selected_features = features["selected_feature_indices"]
		print(f"Selected features: {selected_features}")
		prefix = config["feature_prefix"]
		config["selected_features"] = [ f"{prefix}{i}" \
			for i in selected_features]
	train_evaluate(data_path, output_path, autoencoder_path,
		config, fast_mode)

start_time = time.time()
main()
end_time = round(time.time() - start_time, 2)
print(f"Total time taken: {end_time} seconds")
