import argparse, os, json, joblib, torch
from utils import regression, dataset, autoencoder

X_SKIP_COLS = ["date", "weight", "ts_id", "resp", "resp_1", "resp_2", "resp_3", "resp_4"]
Y_OUTPUT_COLS = ["date", "ts_id"]
METRICS_INFO = ["mse", "r2", "mae"]

def get_cols_for_approach(approach):
	if approach == 1:
		return ["resp_1", "resp_2", "resp_3", "resp_4"]
	if approach == 2:
		return ["resp"]
	if approach == 3:
		return ["resp", "resp_1", "resp_2", "resp_3", "resp_4"]
	print("Invalid approach value: {}".format(approach))
	exit()

def load_model(model_path):
	model = torch.load(model_path)
	model.eval()
	return model

def prepare_data(data_folder, model_path, config, fast_mode):
	model = load_model(model_path)
	y_cols = get_cols_for_approach(config["approach"])
	(train, test, na_value) = dataset.read_data(data_folder,
		fast_mode)
	x_train = train.drop(X_SKIP_COLS, axis=1)
	y_train = train[y_cols]
	x_test = test.drop(X_SKIP_COLS, axis=1)
	y_test = test[y_cols]
	out_train = train[Y_OUTPUT_COLS]
	out_test = test[Y_OUTPUT_COLS]

	print("Encoding data...")
	x_train = autoencoder.encode(model, x_train,
		config["autoencoder_output_features"])
	x_test = autoencoder.encode(model, x_test,
		config["autoencoder_output_features"])
	return (x_train, x_test, y_train, y_test, out_train, out_test, na_value)

def postprocess_data(out_data, y_pred, config):
	y_cols = get_cols_for_approach(config["approach"])
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

def train_evaluate(data_folder, output_folder, autoencoder_path,
	config, fast_mode):
	model = regression.get_model(config["regression_algo"])
	print("Preparing data...")
	(x, x_test, y, y_test, out, out_test, na_value) = prepare_data(
		data_folder, autoencoder_path, config, fast_mode)
	print("Training...")
	model = regression.train(model, x, y)
	print("Evaluating...")
	(y_pred, metrics) = regression.evaluate(model, x_test, y_test,
		METRICS_INFO)
	print("Postprocessing data...")
	y_output = postprocess_data(out_test, y_pred, config)

	output_path = os.path.join(output_folder, "pred.csv")
	y_output.to_csv(output_path, index=False)

	result = { "metrics": metrics, "input_config": config,
		"fast_mode": fast_mode, "na_value": na_value }
	result_path = os.path.join(output_folder, "result.json")
	json_config = json.dumps(result, indent=4)
	with open(result_path, "w") as result_file:
		result_file.write(json_config)

	model_path = os.path.join(output_folder, "{}-approach-{}.joblib".format(
		config["regression_algo"], config["approach"]))
	joblib.dump(model, model_path)
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
		"--autoencoder_path", type=str, help="specifies the autoencoder model path",
		required=True)
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
	autoencoder_path = os.path.abspath(args["autoencoder_path"])
	fast_mode = args["fast_mode"]
	with open(config_path, "r") as json_file:
		config = json.load(json_file)
	train_evaluate(data_path, output_path, autoencoder_path,
		config, fast_mode)

main()
