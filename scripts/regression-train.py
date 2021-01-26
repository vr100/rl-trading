import argparse, os, json, joblib
from utils import regression, dataset

X_SKIP_COLS = ["date", "weight", "ts_id", "resp", "resp_1", "resp_2", "resp_3", "resp_4"]
Y_COLS = ["resp_1", "resp_2", "resp_3", "resp_4"]
Y_COLS_SINGLE = ["resp"]
Y_OUTPUT_COLS = ["date", "ts_id"]
METRICS_INFO = ["mse", "r2", "mape"]

def prepare_data(data_folder, output_type):
	y_cols = Y_COLS_SINGLE if output_type == "single" else Y_COLS
	(train, test, na_value) = dataset.read_data(data_folder)
	x_train = train.drop(X_SKIP_COLS, axis=1)
	y_train = train[y_cols]
	x_test = test.drop(X_SKIP_COLS, axis=1)
	y_test = test[y_cols]
	out_train = train[Y_OUTPUT_COLS]
	out_test = test[Y_OUTPUT_COLS]
	return (x_train, x_test, y_train, y_test, out_train, out_test, na_value)

def postprocess_data(out_data, y_pred, output_type):
	y_cols = Y_COLS_SINGLE if output_type == "single" else Y_COLS
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

def train_evaluate(data_folder, output_folder, reg_type, output_type):
	model = regression.get_model(reg_type)
	print("Preparing data...")
	(x, x_test, y, y_test, out, out_test, na_value) = prepare_data(
		data_folder, output_type)
	print("Training...")
	model = regression.train(model, x, y)
	print("Evaluating...")
	(y_pred, metrics) = regression.evaluate(model, x_test, y_test,
		METRICS_INFO)
	print("Postprocessing data...")
	y_output = postprocess_data(out_test, y_pred, output_type)

	output_path = os.path.join(output_folder, "pred.csv")
	y_output.to_csv(output_path, index=False)

	result = { "metrics": metrics, "na_value": na_value,
		"reg_type": reg_type, "output_type": output_type }
	result_path = os.path.join(output_folder, "result.json")
	json_config = json.dumps(result, indent=4)
	with open(result_path, "w") as result_file:
		result_file.write(json_config)

	model_path = os.path.join(output_folder, "{}-{}.joblib".format(
		reg_type, output_type))
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
		"--output_type", type=str, help="choose single or multi regression",
		required=True, choices=["single", "multi"])
	parser.add_argument(
		"--reg_type", type=str, help="choose regression model",
		required=True, choices=["linear", "kneighbor", "decisiontree",
		"randomforest", "mlp"])
	return vars(parser.parse_args())

def main():
	args = parse_args()
	print("Args: {}".format(args))
	data_path = os.path.abspath(args["data_path"])
	output_path = os.path.abspath(args["output_path"])
	output_type = args["output_type"]
	reg_type = args["reg_type"]
	train_evaluate(data_path, output_path, reg_type, output_type)

main()
