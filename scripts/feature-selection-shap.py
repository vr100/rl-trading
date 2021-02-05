import os, argparse, json, shap
import pandas as pd
import numpy as np
from utils import dataset
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def prepare_data(data_folder, fast_mode, random_mode, config):
	(train, test, na_value) = dataset.read_data(data_folder,
		fast_mode=fast_mode, na_value=config["na_value"],
		random_mode=random_mode)
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

def select_feature(data_folder, output_folder, config, fast_mode,
	random_mode):
	print("Preparing data...")
	(train, test, na_value) = prepare_data(data_folder, fast_mode,
		random_mode, config)
	print("Training model...")
	model = RandomForestRegressor(n_jobs=config["job_count"],
			max_depth=config["max_depth"],
			random_state=config["random_state"])
	model.fit(train["x"], train["y"])
	print("Running feature selection...")
	explainer = shap.TreeExplainer(model, train["x"])
	shap_values = explainer(train["x"])
	print("Getting the plots...")
	plot_path = os.path.join(output_folder, "summary-bar.jpg")
	shap.summary_plot(shap_values, train["x"], plot_type="bar",
		max_display=shap_values.shape[1])
	plt.savefig(plot_path, format="jpg", dpi=200, bbox_inches="tight")
	plt.close()
	plot_path = os.path.join(output_folder, "summary.jpg")
	shap.summary_plot(shap_values, train["x"],
		max_display=shap_values.shape[1])
	plt.savefig(plot_path, format="jpg", dpi=200, bbox_inches="tight")
	plt.close()

	for index in range(shap_values.shape[1]):
		plot_path = os.path.join(output_folder,
			"index-{}.jpg".format(index))
		shap.plots.scatter(shap_values[:,index])
		plt.savefig(plot_path, format="jpg", dpi=200,
			bbox_inches="tight")
		plt.close()

	params = config.copy()
	params["fast_mode"] = fast_mode
	params["random_mode"] = random_mode
	params_path = os.path.join(output_folder, "params.json")
	json_config = json.dumps(params, indent=4)
	with open(params_path, "w") as params_file:
		params_file.write(json_config)

	values = np.absolute(shap_values.values).mean(axis=0)
	result = { 
		"values": values.tolist(),
		"sorted_indices": np.argsort(values).tolist()
	}
	result_path = os.path.join(output_folder, "result.json")
	json_result = json.dumps(result, indent=4)
	with open(result_path, "w") as result_file:
		result_file.write(json_result)

	print("Saved all plots to {}".format(output_folder))

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
	parser.add_argument(
		"--random_mode", default=0, type=int,
		help="specifies the random sample count")
	return vars(parser.parse_args())

def main():
	args = parse_args()
	print("Args: {}".format(args))
	data_path = os.path.abspath(args["data_path"])
	output_path = os.path.abspath(args["output_path"])
	config_path = os.path.abspath(args["config_path"])
	random_mode = args["random_mode"]
	fast_mode = args["fast_mode"]
	with open(config_path, "r") as json_file:
		config = json.load(json_file)
	select_feature(data_path, output_path, config, fast_mode,
		random_mode)

main()
