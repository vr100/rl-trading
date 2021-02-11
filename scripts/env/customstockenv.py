import gym, math
from gym import spaces
import numpy as np

class CustomStockEnv(gym.Env):
	metadata = { "render.modes": ["human"] }

	def __init__(self, data, config):
		super(CustomStockEnv, self).__init__()
		self.data = data
		self.features = config["feature_cols"]
		self.weight = config["weight_col"]
		self.response = config["response_col"]
		self.episode = config["episode_col"]
		self.day_count = config["total_days"]
		# computed reward states: pp_sum, pp_2_sum, p_old, u_old
		# given reward states: weight
		reward_state = 4 + 1
		state_len = len(self.features) + reward_state
		self.action_space = spaces.Discrete(config["total_actions"])
		self.observation_space = spaces.Box(low=-np.inf,
			high=np.inf, shape=(state_len, 1), dtype=np.float32)
		self.pp_sum = 0
		self.pp_2_sum = 0
		self.p_old = 0
		self.u_old = 0
		self.current_step = 0

	def _next_observation(self):
		state_columns = self.features.copy()
		state_columns.append(self.weight)
		state = self.data.iloc[self.current_step][state_columns].to_numpy()
		state = np.append(state, [self.pp_sum, self.pp_2_sum,
			self.p_old, self.u_old])
		state = state.reshape((state.shape[0], 1))
		return state

	def is_done(self):
		if (self.current_step + 1) >= len(self.data):
			return True
		done = self.data.iloc[self.current_step][self.episode] != \
			self.data.iloc[self.current_step + 1][self.episode]
		return done

	def compute_u(self, p_sum, p_2_sum):
		if p_sum != 0:
			t = (p_sum / math.sqrt(p_2_sum)) * math.sqrt(250 / self.day_count)
		else:
			t = 0
		u = min(max(t, 0), 6) * p_sum
		return u

	def _do_nothing(self):
		return 0

	def _perform_action(self):
		current_data = self.data.iloc[self.current_step]
		return_value = current_data[self.weight] * current_data[self.response]
		p = self.p_old + return_value
		p_sum = self.pp_sum + p
		p_2_sum = self.pp_2_sum + pow(p, 2)
		u = self.compute_u(p_sum, p_2_sum)
		reward = u - self.u_old
		done = self.is_done()
		self.p_old = p
		self.u_old = u
		return reward

	def _take_action(self, action):
		if action == 0:
			return self._do_nothing()
		elif action == 1:
			return self._perform_action()
		print("Unknown action: {}".format(action))
		exit()

	def step(self, action):
		reward = self._take_action(action)
		done = self.is_done()
		self.current_step += 1
		if self.current_step >= len(self.data):
			self.current_step = 0
		if done:
			self.reset()
		obs = self._next_observation()
		return obs, reward, done, {}

	def reset(self):
		self.pp_sum = self.pp_sum + self.p_old
		self.pp_2_sum = self.pp_2_sum + pow(self.p_old, 2)
		self.p_old = 0
		self.u_old = 0
		return self._next_observation()

	def render(self, mode="human", close=False):
		u = self.compute_u(self.pp_sum, self.pp_2_sum)
		print(f"pp_sum: {self.pp_sum}, " +
			f"pp_2_sum: {self.pp_2_sum}, " +
			f"p_old: {self.p_old}, " +
			f"u_old: {self.u_old}, " +
			f"computed_u: {u}, " +
			f"current_step: {self.current_step}")
