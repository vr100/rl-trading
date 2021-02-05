import pandas as pd
import os, random, math, csv
from pandas.core.series import Series

# mode to read just top few entries of the data file
FAST_MODE = False
TRAIN_ROWS = 100000
TEST_ROWS = 20000

RANDOM_SECTION_LENGTH = 50000

def compute_step(random_sections, random_count):
	if random_sections <= 0:
		return 0
	step = 1
	element_per_section = random_count / random_sections
	while element_per_section < 1:
		step = step * 2
		actual_sections = math.floor(random_sections / step)
		element_per_section = random_count / actual_sections
	return step

def get_count(sections, step, random_count):
	actual_sections = math.floor(sections / step)
	if actual_sections == 0:
		actual_sections = 1
	element_per_section = random_count / actual_sections
	floor_elements = math.floor(element_per_section)
	# if decimal value >= .5 choose ceil, else floor
	if math.floor(element_per_section * 2) > math.floor(floor_elements * 2):
		return math.ceil(element_per_section)
	else:
		return floor_elements

def get_random_section_index(index, step):
	if step == 1:
		return index
	return index + random.randrange(step)

def get_record_count(file_path):
	with open(file_path, "r") as csv_file:
		sniffer = csv.Sniffer()
		has_header = sniffer.has_header(csv_file.readline())
		if not has_header:
			csv_file.seek(0)
		record_count = sum(1 for line in csv_file)
		print("has header: {}, record count: {}".format(
			has_header, record_count))
		return (has_header, record_count)
	return (False, 0)

def read_random_data(file_path, random_count,
	section_length=RANDOM_SECTION_LENGTH):
	(has_header, record_count) = get_record_count(file_path)
	random_sections = math.floor(record_count / section_length)
	step = compute_step(random_sections, random_count)
	total_count = 0
	random_data = pd.DataFrame()
	print("Sections: {}, Step: {}".format(random_sections, step))
	for index in range(0, random_sections, step):
		random_index = get_random_section_index(index, step)
		skiprows = random_index * section_length
		if has_header:
			skiprows = range(1, skiprows + 1)
		data = pd.read_csv(file_path, skiprows=skiprows,
			nrows=section_length)
		section_count = get_count(random_sections - index,
			step, random_count - total_count)
		total_count = total_count + section_count
		random_list = random.sample(range(0, len(data)), section_count)
		chosen_data = data.iloc[random_list].copy()
		random_data = random_data.append(chosen_data, ignore_index=True)
	print("Required length: {}, actual length: {}".format(
		random_count, len(random_data)))
	return random_data

def read_data(data_folder, fast_mode=FAST_MODE, na_value=None,
	random_mode=0):
	train_path = os.path.join(data_folder, "train.csv")
	if random_mode > 0:
		train = read_random_data(train_path, random_mode)
	elif fast_mode:
		train = pd.read_csv(train_path, nrows=TRAIN_ROWS)
	else:
		train = pd.read_csv(train_path)

	test_path = os.path.join(data_folder, "test.csv")
	if random_mode > 0:
		test_count = math.ceil(0.2 * random_mode)
		test = read_random_data(test_path, test_count)
	elif fast_mode:
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
