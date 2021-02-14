from .customstockenv import CustomStockEnv
import joblib
from utils import regression, scaler
import pandas as pd

class CustomStockEnvPred(CustomStockEnv):

	def __init__(self, data, config):
		super(CustomStockEnvPred, self).__init__(data, config)
		self.prediction = True
		self.model = joblib.load(config["model_path"])
		self.x_scaler = None
		if "x_scaler_path" in config:
			self.x_scaler = joblib.load(config["x_scaler_path"])
		self.y_scaler = None
		if "y_scaler_path" in config:
			self.y_scaler = joblib.load(config["y_scaler_path"])
		self.config = config

	def _predict_response(self, data_row):
		data_row = data_row[self.features]
		data = pd.DataFrame()
		data = data.append(data_row, ignore_index=True)
		if self.x_scaler is not None:
			data, _ = scaler.scale_data(data, self.x_scaler)
		pred = regression.infer(self.model, data, self.config)
		if self.y_scaler is not None:
			pred, _ = scaler.scale_data(pred, self.y_scaler)
		return pred[0][0]
