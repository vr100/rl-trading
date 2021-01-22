import argparse, os, joblib, json, torch
import pandas as pd
from utils import regression, dataset, lstm

PREDICT_X_SKIP_COLS = ["date", "weight", "ts_id", "resp", "resp_1", "resp_2", "resp_3", "resp_4"]
X_COLS = ["resp_1", "resp_2", "resp_3", "resp_4"]
Y_OUTPUT_COLS = ["date", "ts_id"]
Y_COL = ["resp"]
METRICS_INFO = ["mse", "r2", "mape"]
DROPOUT = 0.25
HIDDEN_SIZE = 20

def get_prediction_data(data, model_path):
	x = data.drop(PREDICT_X_SKIP_COLS, axis=1)
	y = data[X_COLS]
	model = joblib.load(model_path)
	(y_pred, metrics) = regression.evaluate(model, x, y, METRICS_INFO)
	y_pred = pd.DataFrame(data=y_pred, columns=X_COLS)
	return (y_pred, metrics)

def prepare_data(data_folder, model_path):
	(train, test, na_value) = dataset.read_data(data_folder, "ts_id")
	x_train = train[X_COLS]
	y_train = train[Y_COL]
	x_test = test[X_COLS]
	y_test = test[Y_COL]
	out_train = train[Y_OUTPUT_COLS]
	out_test = test[Y_OUTPUT_COLS]
	(x_pred_train , metrics_train) = get_prediction_data(train, model_path)
	(x_pred_test, metrics_test) = get_prediction_data(test, model_path)
	train = { "x": x_train, "y": y_train, "x_pred": x_pred_train, "out": out_train}
	test = { "x": x_test, "y": y_test, "x_pred": x_pred_test, "out": out_test}
	metrics = {
		"reg_train_pred": metrics_train,
		"reg_test_pred": metrics_test
	}
	return (train, test, metrics, na_value)

def postprocess_data(out_data, y_pred):
	y_output = out_data.copy()
	y_output[Y_COL] = y_pred
	return y_output

def train_evaluate(data_folder, output_folder, model_path):
	model = lstm.get_model(DROPOUT, len(X_COLS), HIDDEN_SIZE)

	print("Preparing data...")
	(train, test, metrics, na_value) = prepare_data(data_folder, model_path)

	print("Training...")
	model = lstm.train(model, train["x"], train["y"])
	model = lstm.train(model, train["x_pred"], train["y"])

	print("Evaluating...")
	(y_pred, metrics_lstm) = lstm.evaluate(model, test["x"],
		test["y"], METRICS_INFO)
	(y_pred_reg, metrics_reg_lstm) = lstm.evaluate(model,
		test["x_pred"], test["y"], METRICS_INFO)
	metrics["lstm_pred"] = metrics_lstm
	metrics["reg_lstm_pred"] = metrics_reg_lstm

	print("Postprocessing data...")
	y_output = postprocess_data(test["out"], y_pred)
	y_output_reg = postprocess_data(test["out"], y_pred_reg)

	output_path = os.path.join(output_folder, "pred.csv")
	y_output.to_csv(output_path, index=False)

	output_path = os.path.join(output_folder, "pred_reg.csv")
	y_output_reg.to_csv(output_path, index=False)

	result = { "metrics": metrics, "na_value": na_value }
	result_path = os.path.join(output_folder, "result.json")
	json_config = json.dumps(result, indent=4)
	with open(result_path, "w") as result_file:
		result_file.write(json_config)

	model_path = os.path.join(output_folder, "lstm.mdl")
	torch.save(model, model_path)
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
		"--regression_model_path", type=str, required = True,
		help="specifies the regression model path")
	return vars(parser.parse_args())

def main():
	args = parse_args()
	print("Args: {}".format(args))
	data_path = os.path.abspath(args["data_path"])
	output_path = os.path.abspath(args["output_path"])
	model_path = os.path.abspath(args["regression_model_path"])
	train_evaluate(data_path, output_path, model_path)

main()
