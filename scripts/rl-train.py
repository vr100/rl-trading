import os, argparse, json
from utils import a2c, dataset

def prefill_config(data, config):
	feature_cols = [col for col in data if col.startswith(
		config["feature_prefix"])]
	config["feature_cols"] = feature_cols
	episode = config["episode_col"]
	config["total_days"] = len(data[episode].unique())
	return config

def prepare_data(data_folder, fast_mode, random_mode, config):
	(train, test, na_value) = dataset.read_data(data_folder,
		fast_mode=fast_mode, na_value=config["na_value"],
		random_mode=random_mode)
	return (train, test, na_value)

def train_rl(data_folder, output_folder, config, fast_mode,
	random_mode):
	print("Preparing data...")
	(train, test, na_value) = prepare_data(data_folder, fast_mode,
		random_mode, config)
	config = prefill_config(train, config)
	print("Training the model...")
	(model, env) = a2c.get_model(train, config)
	model = a2c.train(model, env, len(train))
	print("Evaluating the model...")
	test = test.sort_values(by=[config["episode_col"]])
	config = prefill_config(test, config)
	a2c.evaluate(model, test, config)
	print("Saving model...")
	output_path = os.path.join(output_folder, "a2c.zip")
	a2c.save(model, output_path)
	print(f"Model saved to {output_path}")

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
	train_rl(data_path, output_path, config, fast_mode,
		random_mode)

main()