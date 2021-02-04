import os, argparse, json
import pandas as pd
from boruta import BorutaPy
from utils import dataset
from sklearn.ensemble import RandomForestRegressor

def prepare_data(data_folder, fast_mode, config):
	(train, test, na_value) = dataset.read_data(data_folder,
		fast_mode=fast_mode, na_value=config["na_value"])
	x_train = train.drop(config["x_skip_columns"], axis=1).to_numpy()
	x_test = test.drop(config["x_skip_columns"], axis=1).to_numpy()
	y_train = train[config["y_columns"]].to_numpy()
	y_test = test[config["y_columns"]].to_numpy()
	if len(config["y_columns"]) == 1:
		y_train = y_train.ravel()
		y_test  = y_test.ravel()
	train = { "x": x_train, "y": y_train }
	test = { "x": x_test, "y": y_test }
	return (train, test, na_value)

def select_feature(data_folder, output_folder, config, fast_mode):
	print("Preparing data...")
	(train, test, na_value) = prepare_data(data_folder, fast_mode,
		config)
	print("Running feature selection...")
	model = RandomForestRegressor(n_jobs=config["job_count"],
			max_depth=config["max_depth"],
			random_state=config["random_state"])
	feature_selector =  BorutaPy(model, n_estimators="auto",
		verbose=2, random_state=config["random_state"],
		max_iter=config["max_iter"])
	print("Fitting feature selection...")
	feature_selector.fit(train["x"], train["y"])
	print("Getting the result...")
	result = {
		"selected_count": feature_selector.n_features_,
		"selected_features": feature_selector.support_.tolist(),
		"weak_features": feature_selector.support_weak_.tolist(),
		"ranking": feature_selector.ranking_.tolist(),
		"importance_history": feature_selector.importance_history_.tolist()
	}
	result_path = os.path.join(output_folder, "result.json")
	json_result = json.dumps(result, indent=4)
	with open(result_path, "w") as result_file:
		result_file.write(json_result)
	print("Result: {}".format(json_result))
	print("Result saved to {}".format(result_path))

	x_train = feature_selector.transform(train["x"])
	print(x_train[20:])


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
	fast_mode = args["fast_mode"]
	with open(config_path, "r") as json_file:
		config = json.load(json_file)
	select_feature(data_path, output_path, config, fast_mode)

main()
