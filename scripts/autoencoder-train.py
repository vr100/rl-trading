from utils import autoencoder, dataset, hyperparams
import argparse, os, torch, json, uuid

X_SKIP_COLS = ["date", "weight", "ts_id", "resp", "resp_1", "resp_2", "resp_3", "resp_4"]
METRICS_INFO = ["mse", "r2", "mae"]

def prepare_data(data_folder, fast_mode, na_value):
	(train, test, na_value) = dataset.read_data(data_folder,
		fast_mode=fast_mode, na_value=na_value)
	x_train = train.drop(X_SKIP_COLS, axis=1)
	x_test = test.drop(X_SKIP_COLS, axis=1)
	return (x_train, x_test, na_value)

def train_evaluate_fn(data_folder, output_folder, fast_mode, params):
	output_folder = os.path.join(output_folder, uuid.uuid4().hex)
	os.mkdir(output_folder)
	print("Preparing data...")
	(train, test, na_value) = prepare_data(data_folder, fast_mode,
		params["na_value"])
	model = autoencoder.get_model(train.shape[1],
		params["expected_features"], params["dropout"])
	model.print_details()
	print("Training...")
	model = autoencoder.train(model, train, epochs=params["epochs"],
		lr=params["lr"])
	print("Evaluating...")
	metrics = autoencoder.evaluate(model, train, METRICS_INFO)

	print("Saving files...")
	result = { "metrics": metrics, "na_value": na_value}
	result_path = os.path.join(output_folder, "result.json")
	json_config = json.dumps(result, indent=4)
	with open(result_path, "w") as result_file:
		result_file.write(json_config)
	params_path = os.path.join(output_folder, "params.json")
	json_config = json.dumps(params, indent=4)
	with open(params_path, "w") as params_file:
		params_file.write(json_config)
	model_path = os.path.join(output_folder, "autoencoder.mdl")
	torch.save(model, model_path)
	print("Output files (model, result) saved to {}".format(
		output_folder))

	data = { "loss": metrics["mse"], "metrics": metrics,
		"result": output_folder, "input": params}
	return data

def train_evaluate(data_folder, output_folder, params_path,
	fast_mode):
	(trials, _) = hyperparams.train_and_evaluate(
		data_folder, output_folder, params_path, fast_mode,
		train_evaluate_fn)

	result = { "best_trial": trials.best_trial["result"],
		"all_trials": trials.results}
	result_path = os.path.join(output_folder, "result.json")
	json_config = json.dumps(result, indent=4)
	with open(result_path, "w") as result_file:
		result_file.write(json_config)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--data_path", type=str, help="specifies the data folder path",
		required=True)
	parser.add_argument(
		"--output_path", type=str, help="specifies the output folder path",
		required=True)
	parser.add_argument(
		"--hyperparams_path", type=str, help="specifies the hyperparams config path",
		default=None)
	parser.add_argument(
		"--params_path", type=str, help="specifies the params config path",
		default=None)
	parser.add_argument(
		"--fast_mode", default=False,
		type=lambda s: s.lower() in ['true', 'yes', '1'],
		help="specifies whether only a sample subset should be run")
	return (vars(parser.parse_args()), parser)

def main():
	(args, parser) = parse_args()
	print("Args: {}".format(args))
	data_path = os.path.abspath(args["data_path"])
	output_path = os.path.abspath(args["output_path"])
	fast_mode = args["fast_mode"]
	if args["hyperparams_path"]:
		hyperparams_path = os.path.abspath(args["hyperparams_path"])
		train_evaluate(data_path, output_path, hyperparams_path,
			fast_mode)
	elif args["params_path"]:
		params_path = os.path.abspath(args["params_path"])
		with open(params_path, "r") as json_file:
			params = json.load(json_file)
		train_evaluate_fn(data_path, output_path, fast_mode, params)
	else:
		print("Either --hyperparams_path or --params_path should be present")
		parser.print_help()

main()
