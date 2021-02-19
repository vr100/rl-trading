import os, argparse, json, time
from utils import rl, dataset
import numpy as np

def prefill_config(data, config):
	if "feature_cols" not in config:
		feature_cols = [col for col in data if col.startswith(
			config["feature_prefix"])]
		config["feature_cols"] = feature_cols
	episode = config["episode_col"]
	config["total_days"] = len(data[episode].unique())
	print(f"Filled Config: {config}")
	return config

def save_config(output_path, config):
	new_config = config.copy()
	new_config.pop("total_days")
	json_config = json.dumps(new_config, indent=4)
	with open(output_path, "w") as config_file:
		config_file.write(json_config)

def prepare_data(data_folder, fast_mode, random_mode, config):
	(train, test, na_value) = dataset.read_data(data_folder,
		fast_mode=fast_mode, na_value=config["na_value"],
		random_mode=random_mode)
	return (train, test, na_value)

def get_result(action_probs, u):
	actions = np.argmax(action_probs, axis=1)
	action_0 = len(list(filter(lambda x: x == 0, actions)))
	action_1 = len(list(filter(lambda x: x == 1, actions)))
	return {"action_0": action_0, "action_1": action_1, "u": u}

def get_action_match_ratio(action_probs, pred_action_probs):
	eval_actions = np.argmax(action_probs, axis=1)
	pred_actions = np.argmax(pred_action_probs, axis=1)
	same_action = [int(item[0] == item[1]) \
		for item in zip(eval_actions, pred_actions)]
	same_action_ratio = len(same_action) / len(eval_actions)
	return same_action_ratio

def train_rl(data_folder, output_folder, config, fast_mode,
	random_mode):
	print("Preparing data...")
	(train, test, na_value) = prepare_data(data_folder, fast_mode,
		random_mode, config)
	config = prefill_config(train, config)
	print("Training the model...")
	(model, env) = rl.get_model(train, config)
	model = rl.train(model, env, config["repeat_train"] * len(train))
	print("Saving model...")
	if not os.path.exists(output_folder):
		print(f"folder: {output_folder} does not exist, creating...")
		os.mkdir(output_folder)
	model_name = config["model"]
	output_path = os.path.join(output_folder, f"{model_name}.zip")
	rl.save(model, output_path)
	print(f"Model saved to {output_path}")
	print("Loading model...")
	model = rl.load(output_path, config)
	print("Evaluating the model...")
	test = test.sort_values(by=[config["episode_col"]])
	config = prefill_config(test, config)
	(action_probs, u) = rl.evaluate(model, test, config)
	print("Evaluating the model with prediction ...")
	(pred_action_probs, pred_u) = rl.evaluate(model, test, config,
		predict=True)
	print("Evaluation with predict actions ...")
	(next_action_probs, next_u) = rl.evaluate(model, test, config,
		given_actions=pred_action_probs)
	output_path = os.path.join(output_folder, "config.json")
	save_config(output_path, config)
	eval_result = get_result(action_probs, u)
	pred_result = get_result(pred_action_probs, pred_u)
	next_eval_result = get_result(next_action_probs, next_u)
	same_action_ratio = get_action_match_ratio(action_probs, pred_action_probs)
	result = { "datalen": len(test), "eval": eval_result,
		"pred": pred_result, "eval_with_pred": next_eval_result,
		"action_match_ratio": same_action_ratio }
	output_path = os.path.join(output_folder, "result.json")
	json_result = json.dumps(result, indent=4)
	with open(output_path, "w") as result_file:
		result_file.write(json_result)
	print("Done...")

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
	print(f"Config: {config}")

	if args["features_path"] is not None:
		features_path = os.path.abspath(args["features_path"])
		with open(features_path, "r") as json_file:
			features = json.load(json_file)
		selected_features = features["selected_feature_indices"]
		print(f"Selected features: {selected_features}")
		prefix = config["feature_prefix"]
		config["feature_cols"] = [ f"{prefix}{i}" for i in selected_features]

	train_rl(data_path, output_path, config, fast_mode,
		random_mode)

start_time = time.time()
main()
end_time = round(time.time() - start_time, 2)
print(f"Total time taken: {end_time} seconds")
