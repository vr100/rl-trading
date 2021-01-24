from utils import autoencoder
from utils import dataset
import argparse, os, torch, json

X_SKIP_COLS = ["date", "weight", "ts_id", "resp", "resp_1", "resp_2", "resp_3", "resp_4"]
EXPECTED_FEATURES = 50
DROPOUT = 0.25
METRICS_INFO = ["mse", "r2", "mape"]

def prepare_data(data_folder, fast_mode):
	(train, test, na_value) = dataset.read_data(data_folder, "ts_id",
		fast_mode=fast_mode, na_value=0)
	x_train = train.drop(X_SKIP_COLS, axis=1)
	x_test = test.drop(X_SKIP_COLS, axis=1)
	return (x_train, x_test, na_value)

def train_evaluate(data_folder, output_folder, fast_mode):
	print("Preparing data...")
	(train, test, na_value) = prepare_data(data_folder, fast_mode)
	model = autoencoder.get_model(train.shape[1], EXPECTED_FEATURES,
		DROPOUT)
	model.print_details()
	print("Training...")
	model = autoencoder.train(model, train)
	print("Evaluating...")
	metrics = autoencoder.evaluate(model, train, METRICS_INFO)
	print("Saving files...")
	result = { "metrics": metrics, "na_value": na_value}
	result_path = os.path.join(output_folder, "result.json")
	json_config = json.dumps(result, indent=4)
	with open(result_path, "w") as result_file:
		result_file.write(json_config)
	model_path = os.path.join(output_folder, "autoencoder.mdl")
	torch.save(model, model_path)
	print("Output files (model, result) saved to {}".format(
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
		"--fast_mode", default=False,
		type=lambda s: s.lower() in ['true', 'yes', '1'],
		help="specifies whether only a sample subset should be run")
	return vars(parser.parse_args())

def main():
	args = parse_args()
	print("Args: {}".format(args))
	data_path = os.path.abspath(args["data_path"])
	output_path = os.path.abspath(args["output_path"])
	fast_mode = args["fast_mode"]
	train_evaluate(data_path, output_path, fast_mode)

main()
