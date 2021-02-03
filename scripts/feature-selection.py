import os, argparse, json
import pandas as pd
from BorutaShap import BorutaShap
from utils import dataset

def prepare_data(data_folder, fast_mode, config):
	(train, test, na_value) = dataset.read_data(data_folder,
		fast_mode=fast_mode, na_value=config["na_value"])
	x_train = train.drop(config["x_skip_columns"], axis=1)
	x_test = test.drop(config["x_skip_columns"], axis=1)
	y_train = train[config["y_columns"]]
	y_test = test[config["y_columns"]]
	train = { "x": x_train, "y": y_train }
	test = { "x": x_test, "y": y_test }
	return (train, test, na_value)

def select_feature(data_folder, output_folder, config, fast_mode):
	print("Preparing data...")
	(train, test, na_value) = prepare_data(data_folder, fast_mode,
		config)
	feature_selector = BorutaShap(importance_measure="shap",
		classification=False)
	feature_selector.fit(X=train["x"],
		y=train["y"].to_numpy(),
		n_trials=config["trail_count"],
		random_state=config["random_state"], verbose=True)
	box_plot = feature_selector.plot(which_features="all",
		figsize=(16,12))
	output_path = os.path.join(output_folder, "feature-plot.jpg")
	box_plot.figure.savefig(output_path, format="jpeg", dpi=100)
	subset = feature_selector.Subset()
	print(subset.head())
	print(len(subset))

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
