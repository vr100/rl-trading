import gym
from stable_baselines3 import A2C, PPO, DDPG
from env.customstockenv import CustomStockEnv

SUPPORTED_MODELS = ["A2C", "PPO", "DQN"]

def get_class(model_name):
	check_supported(model_name)
	module = __import__("stable_baselines3")
	clz = getattr(module, model_name)
	return clz

def check_supported(model_name):
	if model_name not in SUPPORTED_MODELS:
		print(f"Unknown model {model_name}, supported models: {SUPPORTED_MODELS}")
		exit()

def get_model(data, config):
	env = CustomStockEnv(data, config)
	clz = get_class(config["model"])
	constructor_fn = getattr(clz, "__init__")
	model = clz(policy="MlpPolicy", env=env, verbose=1)
	return (model, env)

def train(model, env, timesteps):
	model = model.learn(total_timesteps=timesteps)
	print("Training result: ")
	env.render()
	return model

def evaluate(model, test, config):
	env = CustomStockEnv(test, config)
	obs = env.reset()
	datalen = len(test)
	no_action = 0
	for i in range(datalen):
		action, stats = model.predict(obs, deterministic=True)
		if action == 0:
			no_action += 1
		obs, reward, done, info = env.step(action)
		if i % config["print_step"] == 0:
			weight = test.iloc[i][config["weight_col"]]
			response = test.iloc[i][config["response_col"]]
			print(f"Weight: {weight}, response: {response}, " +
				f"Action: {action}, Stats: {stats}, Reward: {reward}")
			env.render()
	print("Test result: ")
	env.render()
	print(f"No action taken for data point: {no_action} / {datalen}")

def save(model, output_path):
	model.save(output_path)

def load(model_path, config, env=None, data=None):
	clz = get_class(config["model"])
	load_fn = getattr(clz, "load")
	if env:
		return load_fn(model_path, env)
	if data is not None:
		env = CustomStockEnv(data, config)
		return load_fn(model_path, env)
	return load_fn(model_path, env=None)
