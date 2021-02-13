import gym
from stable_baselines3 import A2C, PPO, DDPG, SAC
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from env.customstockenvdefault import CustomStockEnvDefault
from env.customstockenvpred import CustomStockEnvPred
import numpy as np

DEFAULT_VALUES = {
	"ent_coef": 0.005,
	"sigma": 0.5
}

def get_value(config, name):
	if name not in config:
		return DEFAULT_VALUES[name]
	return config[name]

def get_a2c_model(env, config):
	return A2C("MlpPolicy", env, verbose=1)

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

def evaluate(model, test, config, predict=False):
	if predict and "model_path" in config:
		print("Using prediction for evaluation...")
		env = CustomStockEnvPred(test, config)
	else:
		env = CustomStockEnvDefault(test, config)
	obs = env.reset()
	datalen = len(test)
	no_action = 0
	for i in range(datalen):
		action_prob, stats = model.predict(obs)
		action = np.argmax(action_prob)
		if action == 0:
			no_action += 1
		obs, reward, done, info = env.step(action_prob)
		if i % config["print_step"] == 0:
			weight = test.iloc[i][config["weight_col"]]
			response = test.iloc[i][config["response_col"]]
			print(f"Weight: {weight}, response: {response}, " +
				f"Action: {action}, Stats: {stats}, Reward: {reward}")
			env.render()
	print("Test result: ")
	env.render()
	print(f"No action taken for data point: {no_action} / {datalen}")
	computed_u, _ = env.compute_u()
	return (no_action, computed_u)

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
