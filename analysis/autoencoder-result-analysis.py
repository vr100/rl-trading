import argparse, os, json

def analyze(input_path, output_path, config):
	with open(input_path, "r") as json_file:
		input_data = json.load(json_file)
	all_trials = input_data["all_trials"]
	best_trial = input_data["best_trial"]

	# find all matching trails
	matching_trials = []
	for trial in all_trials:
		matching = True
		for param in config:
			if trial["input"][param] != config[param]:
				matching = False
				break
		if matching:
			matching_trials.append(trial)
	# check if best trial matches
	best_matching = True
	for param in config:
		if best_trial["input"][param] != config[param]:
			best_matching = False
			break
	analysis = {}
	# analyze losses
	losses = []
	for trial in matching_trials:
		losses.append(trial["loss"])
	min_value = min(losses)
	min_index = losses.index(min_value)
	max_value = max(losses)
	max_index = losses.index(max_value)
	mean = sum(losses) / len(losses)
	mid_index = len(losses) // 2
	median = sorted(losses)[mid_index]
	variance = sum((i - mean) ** 2 for i in losses) / len(losses)
	stats = {
		"mean": mean, "median": median, "variance": variance,
		"min": min_value, "min_trial": matching_trials[min_index],
		"max": max_value, "max_trial_id": matching_trials[max_index]
	}
	result = {
		"input": config,
		"matches_best_trial": best_matching,
		"stats": stats,
		"matching_trials": matching_trials
	}
	json_result = json.dumps(result, indent=4)
	with open(output_path, "w") as result_file:
		result_file.write(json_result)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--input_path", type=str, help="specifies the file path for the input result json file",
		required=True)
	parser.add_argument(
		"--output_path", type=str, help="specifies the output file path",
		required=True)
	parser.add_argument(
		"--config_path", type=str, help="specifies the config path",
		required=True)
	return vars(parser.parse_args())

def main():
	args = parse_args()
	print("Args: {}".format(args))
	input_path = os.path.abspath(args["input_path"])
	output_path = os.path.abspath(args["output_path"])
	config_path = os.path.abspath(args["config_path"])
	with open(config_path, "r") as json_file:
		config = json.load(json_file)
	print("Config: {}".format(config))
	analyze(input_path, output_path, config)

main()