import json, os
from hyperopt import hp
from functools import partial
from hyperopt import fmin, Trials, tpe, STATUS_OK

MAX_EVALS = 30

def load_params(params_path):
	with open(params_path, "r") as json_file:
		metadata = json.load(json_file)
	return load_space(metadata)

def handle_nested_choice(name, params):
	choices = []
	for p in params:
		choices.append(load_space(p))
	return hp.choice(name, choices)

def get_space(p, param):
	fn_name = param["function"]
	args = param["args"]
	if fn_name == "nested_choice":
		return handle_nested_choice(p, args)
	fn = getattr(hp, fn_name)
	if fn_name == "choice":
		return fn(p, args)
	return fn(p, *args)

def load_space(params):
	spaces = {}
	for p in params:
		spaces[p] = get_space(p, params[p])
	return spaces

def objective_fn(train_fn, data_folder, output_folder,
		fast_mode, chosen_params):
	data = train_fn(data_folder, output_folder, fast_mode,
		chosen_params)
	data["status"] = STATUS_OK
	return data

def train_and_evaluate(data_folder, output_folder, params_path,
	fast_mode, train_fn):
	hyperparams = load_params(params_path)
	trials = Trials()
	objective = partial(objective_fn, train_fn, data_folder,
		output_folder, fast_mode)
	tpe_best = fmin(fn=objective, space=hyperparams,
		algo=tpe.suggest, trials=trials, max_evals=MAX_EVALS)
	return (trials, tpe_best)
