import pandas as pd
import os, random, math

# mode to read just top few entries of the data file
FAST_MODE = False
TRAIN_ROWS = 10000
TEST_ROWS = 2000

def read_data(data_folder, max_col, fast_mode=FAST_MODE,
	na_value=None):
	train_path = os.path.join(data_folder, "train.csv")
	if fast_mode:
		train = pd.read_csv(train_path, nrows=TRAIN_ROWS)
	else:
		train = pd.read_csv(train_path)
	max_value = train.max()[max_col]

	test_path = os.path.join(data_folder, "test.csv")
	if fast_mode:
		test = pd.read_csv(test_path, nrows=TEST_ROWS)
	else:
		test = pd.read_csv(test_path)
	max_value = max(max_value, test.max()[max_col]) + 1000

	if na_value is None:
		na_value = max_value

	train = train.fillna(na_value)
	test = test.fillna(na_value)

	return (train, test, na_value)

def split_data(data_path, output_folder, time_sensitive):
	full_data = pd.read_csv(data_path)

	min_date = full_data["date"].min()
	max_date = full_data["date"].max() + 1
	test_count = math.floor(0.2 * (max_date - min_date))

	if time_sensitive:
		test_start = max_date - test_count
		test_list = list(range(test_start, max_date))
	else:
		test_list = random.sample(range(min_date, max_date), test_count)

	test_data = full_data[full_data["date"].isin(test_list)]
	train_data = full_data[~full_data["date"].isin(test_list)]

	print("Test len: {}".format(len(test_data)))
	print("Train len: {}".format(len(train_data)))

	test_output = os.path.join(output_folder, "test.csv")
	train_output = os.path.join(output_folder, "train.csv")

	test_data.to_csv(test_output)
	train_data.to_csv(train_output)
	print("Split data saved to {}, {}".format(test_output, train_output))
