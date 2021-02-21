import os, time, argparse, uuid, importlib, json
import utils.hyperparams as hp
from functools import partial

def train_fn(rl_mod, config, data_folder, output_folder, fastmode, params):
	config = config.copy()
	config["params"] = params.copy()
	output_folder = os.path.join(output_folder, uuid.uuid4().hex)
	os.mkdir(output_folder)
	return rl_mod.train_rl(data_folder, output_folder, config,
		fastmode, 0)

def train_rl_hyperopt(data_folder, output_folder, hyperparams_path,
	features_path, config):
	rl_mod = importlib.__import__("rl-train")
	config = rl_mod.update_features(features_path, config)
	train_partial_fn = partial(train_fn, rl_mod, config)
	if not os.path.exists(output_folder):
		print(f"folder: {output_folder} does not exist, creating...")
		os.mkdir(output_folder)
	(trials, _) = hp.train_and_evaluate(data_folder, output_folder,
		hyperparams_path, False, train_partial_fn,
		max_evals=config["max_evals"])
	result = { "best_trial": trials.best_trial["result"],
		"all_trials": trials.results}
	result_path = os.path.join(output_folder, "result.json")
	json_result = json.dumps(result, indent=4)
	with open(result_path, "w") as result_file:
		result_file.write(json_result)

	output_params_path = os.path.join(output_folder,
		"hyperparams.json")
	with open(hyperparams_path, "r") as json_file:
		metadata = json.load(json_file)
	json_metadata = json.dumps(metadata,indent=4)
	with open(output_params_path, "w") as json_file:
		json_file.write(json_metadata)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--data_path", type=str, help="specifies the data folder path",
		required=True)
	parser.add_argument(
		"--output_path", type=str, help="specifies the output folder path",
		required=True)
	parser.add_argument(
		"--hyperparams_path", type=str, help="specifies the json hyperparams path",
		required=True)
	parser.add_argument(
		"--config_path", type=str, help="specifies the base config path",
		required=True)
	parser.add_argument(
		"--features_path", type=str, help="specifies the feature selection results json file path",
		default=None)
	return vars(parser.parse_args())

def main():
	args = parse_args()
	print("Args: {}".format(args))
	data_path = os.path.abspath(args["data_path"])
	output_path = os.path.abspath(args["output_path"])
	hyperparams_path = os.path.abspath(args["hyperparams_path"])
	config_path = os.path.abspath(args["config_path"])
	features_path = args["features_path"]
	with open(config_path, "r") as json_file:
		config = json.load(json_file)
	print(f"Config: {config}")

	train_rl_hyperopt(data_path, output_path, hyperparams_path,
		features_path, config)

def time_wrapped_main():
	start_time = time.time()
	main()
	end_time = round(time.time() - start_time, 2)
	print(f"Total time taken: {end_time} seconds")

if __name__ == "__main__":
	time_wrapped_main()
