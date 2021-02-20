import gym
from stable_baselines3 import A2C, PPO, DDPG, SAC
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from env.customstockenvdefault import CustomStockEnvDefault
from env.customstockenvpred import CustomStockEnvPred
import numpy as np
from functools import partial

DEFAULT_VALUES = {
	"ent_coef": 0.005,
	"sigma": 0.5
}

A2C_DEFAULT_VALUES = {
	"const_lr": 7e-4,
	"var_lr": 0,
	"gamma": 0.99,
	"use_rms_prop": True,
	"gae_lambda": 1.0,
	"seed": None,
	"ent_coef": 0.0,
	"vf_coef": 0.5,
	"use_sde": False,
	"n_steps": 5
}

def get_value(config, name, default=DEFAULT_VALUES):
	if name not in config:
		return default[name]
	return config[name]

def a2c_lr_sched(config, progress):
	const_lr = get_value(config, "const_lr", default=A2C_DEFAULT_VALUES)
	var_lr = get_value(config, "var_lr", default=A2C_DEFAULT_VALUES)
	return const_lr - (1 - progress) * var_lr

def get_a2c_model(env, config):
	params = config["params"]
	lr_fn = partial(a2c_lr_sched, params)
	gamma = get_value(params, "gamma", default=A2C_DEFAULT_VALUES)
	use_rms_prop = get_value(params, "use_rms_prop", default=A2C_DEFAULT_VALUES)
	gae_lambda = get_value(params, "gae_lambda", default=A2C_DEFAULT_VALUES)
	seed = get_value(params, "seed", default=A2C_DEFAULT_VALUES)
	ent_coef = get_value(params, "ent_coef", default=A2C_DEFAULT_VALUES)
	vf_coef = get_value(params, "vf_coef", default=A2C_DEFAULT_VALUES)
	use_sde = get_value(params, "use_sde", default=A2C_DEFAULT_VALUES)
	n_steps = get_value(params, "n_steps", default=A2C_DEFAULT_VALUES)
	return A2C("MlpPolicy", env, verbose=1, learning_rate=lr_fn,
		gamma=gamma, use_rms_prop=use_rms_prop, gae_lambda=gae_lambda,
		seed=seed, ent_coef=ent_coef, vf_coef=vf_coef, use_sde=use_sde,
		n_steps=n_steps)

def get_ppo_model(env, config):
	ent_coef = get_value(config, "ent_coef")
	return PPO("MlpPolicy", env, ent_coef=ent_coef, verbose=1)

def get_ddpg_model(env, config):
	sigma = get_value(config, "sigma")
	action_count = config["total_actions"]
	action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(action_count),
		sigma=float(sigma) * np.ones(action_count))
	return DDPG("MlpPolicy", env, action_noise=action_noise,
		verbose=1)

def get_sac_model(env, config):
	ent_coef = get_value(config, "ent_coef")
	sigma = get_value(config, "sigma")
	action_count = config["total_actions"]
	action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(action_count),
		sigma=float(sigma) * np.ones(action_count))
	return SAC("MlpPolicy", env, ent_coef=ent_coef,
		action_noise=action_noise, verbose=1)

MODEL_FN = {
	"A2C": get_a2c_model,
	"PPO": get_ppo_model,
	"DDPG": get_ddpg_model,
	"SAC": get_sac_model
}

def get_model_fn(model_name):
	if model_name not in MODEL_FN:
		print(f"Unknown model {model_name}, supported models: {SUPPORTED_MODELS}")
		exit()
	return MODEL_FN[model_name]

def check_supported(model_name):
	if model_name not in MODEL_FN:
		print(f"Unknown model {model_name}, supported models: {SUPPORTED_MODELS}")
		exit()

def get_class(model_name):
	check_supported(model_name)
	module = __import__("stable_baselines3")
	clz = getattr(module, model_name)
	return clz

def get_model(data, config):
	env = CustomStockEnvDefault(data, config)
	model_fn = get_model_fn(config["model"])
	model = model_fn(env, config)
	return (model, env)

def train(model, env, timesteps):
	model = model.learn(total_timesteps=timesteps)
	print("Training result: ")
	env.render()
	return model

def evaluate(model, test, config, predict=False,
	given_actions=None):
	if predict and "model_path" in config:
		print("Using prediction for evaluation...")
		env = CustomStockEnvPred(test, config)
	else:
		env = CustomStockEnvDefault(test, config)
	obs = env.reset()
	datalen = len(test)
	action_probs = np.empty(shape=(0, config["total_actions"]))
	for i in range(datalen):
		if given_actions is None:
			action_prob, stats = model.predict(obs)	
		else:
			action_prob = given_actions[i]
			stats = None
		action = np.argmax(action_prob)
		action_probs = np.append(action_probs, action_prob.reshape(
			(1, config["total_actions"])), axis=0)
		obs, reward, done, info = env.step(action_prob)
		if i % config["print_step"] == 0:
			weight = test.iloc[i][config["weight_col"]]
			response = test.iloc[i][config["response_col"]]
			print(f"Weight: {weight}, response: {response}, " +
				f"Action: {action}, Stats: {stats}, Reward: {reward}")
			env.render()
	print("Test result: ")
	env.render()
	actions = np.argmax(action_probs, axis=1)
	no_action = len(list(filter(lambda x: x == 0, actions)))
	print(f"No action taken for data point: {no_action} / {datalen}")
	computed_u, _ = env.compute_u()
	return (action_probs, computed_u)

def save(model, output_path):
	model.save(output_path)

def load(model_path, config, env=None, data=None):
	clz = get_class(config["model"])
	load_fn = getattr(clz, "load")
	if env:
		return load_fn(model_path, env)
	if data is not None:
		env = CustomStockEnvDefault(data, config)
		return load_fn(model_path, env)
	return load_fn(model_path, env=None)
