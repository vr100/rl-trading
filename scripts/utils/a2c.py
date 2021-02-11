import gym
from stable_baselines3 import A2C
from env.customstockenv import CustomStockEnv

def get_model(data, config):
	env = CustomStockEnv(data, config)
	model = A2C("MlpPolicy", env, verbose=1)
	return (model, env)

def train(model, env, timesteps):
	model = model.learn(total_timesteps=timesteps)
	print("Training result: ")
	env.render()
	return model

def evaluate(model, test, config, print_step=1000):
	env = CustomStockEnv(test, config)
	obs = env.reset()
	datalen = len(test)
	no_action = 0
	for i in range(datalen):
		action, stats = model.predict(obs)
		if action == 0:
			no_action += 1
		obs, reward, done, info = env.step(action)
		if i % print_step == 0:
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

def load(model_path, env=None, data=None, config=None):
	if env:
		return A2C.load(model_path, env)
	if data is not None and config is not None:
		env = CustomStockEnv(data, config)
		return A2C.load(model_path, env)
	return A2C.load(model_path, env=None)
