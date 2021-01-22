import argparse, os
from utils import dataset

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--data_path", type=str, help="specifies the train csv data file path",
		required=True)
	parser.add_argument(
		"--output_path", type=str, help="specifies the output folder path",
		required=True)
	parser.add_argument(
		"--time_sensitive", default=False,
		type=lambda s: s.lower() in ['true', 'yes', '1'],
		help="specifies whether the test data should be after train data")
	return vars(parser.parse_args())

def main():
	args = parse_args()
	print("Args: {}".format(args))
	data_path = os.path.abspath(args["data_path"])
	output_path = os.path.abspath(args["output_path"])
	time_sensitive = args["time_sensitive"]
	dataset.split_data(data_path, output_path, time_sensitive)

main()
