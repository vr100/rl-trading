import gym
from stable_baselines3 import A2C
from env.customstockenv import CustomStockEnv

def get_model(data, config):
	env = CustomStockEnv(data, config)
	model = A2C("MlpPolicy", env, verbose=1)
	return (model, env)

def train(model, env, timesteps):
	model = model.learn(total_timesteps=timesteps)
	return model

def evaluate(model, test, config, print_step=100):
	env = CustomStockEnv(test, config)
	obs = env.reset()
	datalen = len(test)
	for i in range(datalen):
		action, stats = model.predict(obs)
		obs, reward, done, info = env.step(action)
		if i % print_step == 0:
			weight = test.iloc[i][config["weight_col"]]
			response = test.iloc[i][config["response_col"]]
			print(f"Weight: {weight}, response: {response}, " +
				f"Action: {action}, Stats: {stats}, Reward: {reward}")
			env.render()
	print("Final result: ")
	env.render()

def save(model, output_path):
	model.save(output_path)

def load(model_path, data, config):
	env = CustomStockEnv(data, config)
	model = A2C.load(model_path, env)
	return model
