import pandas as pd
import os, random, math
from pandas.core.series import Series

# mode to read just top few entries of the data file
FAST_MODE = False
TRAIN_ROWS = 100000
TEST_ROWS = 20000

def read_data(data_folder, fast_mode=FAST_MODE, na_value=None):
	train_path = os.path.join(data_folder, "train.csv")
	if fast_mode:
		train = pd.read_csv(train_path, nrows=TRAIN_ROWS)
	else:
		train = pd.read_csv(train_path)

	test_path = os.path.join(data_folder, "test.csv")
	if fast_mode:
		test = pd.read_csv(test_path, nrows=TEST_ROWS)
	else:
		test = pd.read_csv(test_path)

	if na_value == "zero":
		na_value = 0
	elif na_value == "mean":
		na_value = train.mean()

	if na_value is None:
		na_value = train.mean()

	train = train.fillna(na_value)
	test = test.fillna(na_value)

	if isinstance(na_value,Series):
		na_value = na_value.tolist()

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

	test_data.to_csv(test_output, index=False)
	train_data.to_csv(train_output, index=False)
	print("Split data saved to {}, {}".format(test_output, train_output))
