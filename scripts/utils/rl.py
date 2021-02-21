import gym
from stable_baselines3 import A2C, PPO, DDPG, SAC
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from env.customstockenvdefault import CustomStockEnvDefault
from env.customstockenvpred import CustomStockEnvPred
import numpy as np
from functools import partial

DEFAULT_VALUES = {}

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

PPO_DEFAULT_VALUES = {
	"const_lr": 3e-4,
	"var_lr": 0,
	"gamma": 0.99,
	"gae_lambda": 0.95,
	"const_clip_range": 0.2,
	"var_clip_range": 0.0,
	"const_clip_range_vf": None,
	"var_clip_range_vf": None,
	"ent_coef": 0.0,
	"vf_coef": 0.5,
	"n_steps": 2048,
	"use_sde": False,
	"seed": None
}

DDPG_DEFAULT_VALUES = {
	"const_lr": 1e-3,
	"var_lr": 0,
	"buffer_size": int(1e6),
	"learning_starts": 100,
	"tau": 0.005,
	"gamma": 0.95,
	"optimize_memory_usage": False,
	"seed": None,
	"train_freq": -1,
	"sigma": 0.5
}

SAC_DEFAULT_VALUES = {
	"const_lr": 3e-4,
	"var_lr": 0,
	"buffer_size": int(1e6),
	"learning_starts": 100,
	"tau": 0.005,
	"gamma": 0.99,
	"train_freq": 1,
	"optimize_memory_usage": False,
	"ent_coef": "auto",
	"use_sde": False,
	"target_update_interval": 1,
	"target_entropy": "auto",
	"seed": None,
	"sigma": 0.5
}

def get_value(config, name, default=DEFAULT_VALUES):
	if name not in config:
		return default[name]
	return config[name]

def clip_sched(config, default, progress):
	const_clip = get_value(config, "const_clip_range", default)
	var_clip = get_value(config, "var_clip_range", default)
	return const_clip - (1 - progress) * var_clip

def clip_vf_sched(config, default, progress):
	const_clip = get_value(config, "const_clip_range_vf", default)
	var_clip = get_value(config, "var_clip_range_vf", default)
	return const_clip - (1 - progress) * var_clip

def lr_sched(config, default, progress):
	const_lr = get_value(config, "const_lr", default)
	var_lr = get_value(config, "var_lr", default)
	return const_lr - (1 - progress) * var_lr

def get_a2c_model(env, config):
	params = config["params"]
	lr_fn = partial(lr_sched, params, A2C_DEFAULT_VALUES)
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
	params = config["params"]
	lr_fn = partial(lr_sched, params, PPO_DEFAULT_VALUES)
	const_clip = get_value(params, "const_clip_range", PPO_DEFAULT_VALUES)
	clip_fn = partial(clip_sched, params, PPO_DEFAULT_VALUES) \
		if const_clip is not None else None
	const_clip = get_value(params, "const_clip_range_vf", PPO_DEFAULT_VALUES)
	clip_vf_fn = partial(clip_vf_sched, params, PPO_DEFAULT_VALUES) \
		if const_clip is not None else None
	gamma = get_value(params, "gamma", default=PPO_DEFAULT_VALUES)
	gae_lambda = get_value(params, "gae_lambda", default=PPO_DEFAULT_VALUES)
	ent_coef = get_value(params, "ent_coef", default=PPO_DEFAULT_VALUES)
	vf_coef = get_value(params, "vf_coef", default=PPO_DEFAULT_VALUES)
	n_steps = get_value(params, "n_steps", default=PPO_DEFAULT_VALUES)
	use_sde = get_value(params, "use_sde", default=PPO_DEFAULT_VALUES)
	seed = get_value(params, "seed", default=PPO_DEFAULT_VALUES)
	return PPO("MlpPolicy", env, verbose=1, learning_rate=lr_fn,
		clip_range=clip_fn, clip_range_vf=clip_vf_fn, gamma=gamma,
		gae_lambda=gae_lambda, ent_coef=ent_coef, vf_coef=vf_coef,
		n_steps=n_steps, use_sde=use_sde, seed=seed)

def get_ddpg_model(env, config):
	params = config["params"]
	lr_fn = partial(lr_sched, params, DDPG_DEFAULT_VALUES)
	buffer_size = get_value(params, "buffer_size", default=DDPG_DEFAULT_VALUES)
	learning_starts = get_value(params, "learning_starts", default=DDPG_DEFAULT_VALUES)
	tau = get_value(params, "tau", default=DDPG_DEFAULT_VALUES)
	gamma = get_value(params, "gamma", default=DDPG_DEFAULT_VALUES)
	optimize_memory_usage = get_value(params, "optimize_memory_usage", default=DDPG_DEFAULT_VALUES)
	seed = get_value(params, "seed", default=DDPG_DEFAULT_VALUES)
	train_freq = get_value(params, "train_freq", default=DDPG_DEFAULT_VALUES)
	sigma = get_value(params, "sigma", default=DDPG_DEFAULT_VALUES)
	action_count = config["total_actions"]
	action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(action_count),
		sigma=float(sigma) * np.ones(action_count))
	return DDPG("MlpPolicy", env, action_noise=action_noise,
		verbose=1, learning_rate=lr_fn, buffer_size=buffer_size,
		learning_starts=learning_starts, tau=tau, gamma=gamma,
		optimize_memory_usage=optimize_memory_usage, seed=seed,
		train_freq=train_freq)

def get_sac_model(env, config):
	params = config["params"]
	lr_fn = partial(lr_sched, params, SAC_DEFAULT_VALUES)
	buffer_size = get_value(params, "buffer_size", default=SAC_DEFAULT_VALUES)
	learning_starts = get_value(params, "learning_starts", default=SAC_DEFAULT_VALUES)
	tau = get_value(params, "tau", default=SAC_DEFAULT_VALUES)
	gamma = get_value(params, "gamma", default=SAC_DEFAULT_VALUES)
	train_freq = get_value(params, "train_freq", default=SAC_DEFAULT_VALUES)
	optimize_memory_usage = get_value(params, "optimize_memory_usage", default=SAC_DEFAULT_VALUES)
	ent_coef = get_value(params, "ent_coef", default=SAC_DEFAULT_VALUES)
	use_sde = get_value(params, "use_sde", default=SAC_DEFAULT_VALUES)
	target_update_interval = get_value(params, "target_update_interval", default=SAC_DEFAULT_VALUES)
	target_entropy = get_value(params, "target_entropy", default=SAC_DEFAULT_VALUES)
	seed = get_value(params, "seed", default=SAC_DEFAULT_VALUES)
	sigma = get_value(params, "sigma", default=SAC_DEFAULT_VALUES)
	action_count = config["total_actions"]
	action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(action_count),
		sigma=float(sigma) * np.ones(action_count))
	return SAC("MlpPolicy", env, action_noise=action_noise,
		verbose=1, learning_rate=lr_fn, buffer_size=buffer_size,
		learning_starts=learning_starts, tau=tau, gamma=gamma,
		train_freq=train_freq, optimize_memory_usage=optimize_memory_usage,
		ent_coef=ent_coef, use_sde=use_sde,
		target_update_interval=target_update_interval,
		target_entropy=target_entropy, seed=seed)

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

def get_model(data, config, predict=False):
	if predict and "model_path" in config:
		print("Using prediction for training...")
		env = CustomStockEnvPred(data, config)
	else:
		env = CustomStockEnvDefault(data, config)
	model_fn = get_model_fn(config["model"])
	model = model_fn(env, config)
	return (model, env)

def train(model, env, timesteps):
	model = model.learn(total_timesteps=timesteps)
	print("Training result: ")
	env.render()
	u,_ = env.compute_u()
	return (model, u)

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
