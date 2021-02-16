import argparse, os, json
from utils import dataset
import pandas as pd

def generate_random_data(data_folder, output_folder, config):
	(train, test, na_value) = dataset.read_data(data_folder,
		na_value=config["na_value"],
		random_mode=config["random_count"])
	config["generated_na_value"] = na_value
	output_path = os.path.join(output_folder, "config.json")
	json_config = json.dumps(config, indent=4)
	with open(output_path, "w") as json_file:
		json_file.write(json_config)

	test_output = os.path.join(output_folder, "test.csv")
	train_output = os.path.join(output_folder, "train.csv")

	test.to_csv(test_output, index=False)
	train.to_csv(train_output, index=False)
	print(f"Files saved to folder {output_folder}")

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--data_path", type=str, help="specifies the data folder path",
		required=True)
	parser.add_argument(
		"--output_path", type=str, help="specifies the output folder path",
		required=True)
	parser.add_argument(
		"--random_count", type=int,
		help="specifies whether the size of generated dataset",
		required=True)
	parser.add_argument(
		"--na_value", default="mean", type=str, choices=["mean", "zero"],
		help="specifies the na value")
	return vars(parser.parse_args())

def main():
	args = parse_args()
	print("Args: {}".format(args))
	data_path = os.path.abspath(args["data_path"])
	output_path = os.path.abspath(args["output_path"])
	random_count = args["random_count"]
	na_value = args["na_value"]
	config = { "random_count": random_count, "na_value": na_value }
	generate_random_data(data_path, output_path, config)

main()
