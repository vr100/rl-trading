from .customstockenv import CustomStockEnv

class CustomStockEnvDefault(CustomStockEnv):

	def __init__(self, data, config):
		super(CustomStockEnvDefault, self).__init__(data, config)
		self.prediction = False

	def _predict_response(self, data):
		print("Incorrect flow. Cannot predict for this env...")
		exit()
