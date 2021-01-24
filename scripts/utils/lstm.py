import torch.nn as nn
import torch
from . import metrics as metrics_util
import numpy as np

DEVICE = "cpu"

class CustomLSTMCell(nn.Module):
	def __init__(self, input_size, hidden_size):
		super().__init__()
		self.lstm = nn.LSTMCell(input_size, hidden_size)

	def forward(self, x):
		output = self.lstm(x)
		return output[0]

def get_model(dropout, input_size, hidden_size):
	lstm_layers = [
		nn.Dropout(dropout),
		CustomLSTMCell(input_size, hidden_size),
		nn.Dropout(dropout)
	]
	model = nn.Sequential(
		*lstm_layers,
		nn.Linear(hidden_size, 1)
	)
	return model

def train(model, x, y, epoch=25):
	x = torch.from_numpy(x.to_numpy()).to(DEVICE).float()
	y = torch.from_numpy(y.to_numpy()).to(DEVICE).float()
	loss_fn = nn.L1Loss()
	optimiser = torch.optim.Adam(model.parameters(), lr=0.009)
	start = 0
	batch_size = 64
	for i in range(epoch):
		print("training epoch {}/{}".format(i + 1, epoch))
		while start < len(x):
			batch_len = batch_size if (start + batch_size) < len(x) \
				else (len(x) - start)
			x_batch = torch.narrow(x, 0, start, batch_len)
			y_batch = torch.narrow(y, 0, start, batch_len)

			output = model(x_batch)
			loss = loss_fn(output, y_batch)
			optimiser.zero_grad()
			if (start + batch_len) < len(x):
				loss.backward(retain_graph = True)
			else:
				loss.backward(retain_graph = False)
			optimiser.step()

			start = start + batch_len

	return model

def evaluate(model, x, y, metrics):
	x = torch.from_numpy(x.to_numpy()).to(DEVICE).float()
	y_pred = np.empty(shape=(0, 1))
	with torch.no_grad():
		start = 0
		batch_size = 64
		while start < len(x):
			batch_len = batch_size if (start + batch_size) < len(x) \
				else (len(x) - start)
			x_batch = torch.narrow(x, 0, start, batch_len)
			output = model(x_batch)
			y_pred = np.append(y_pred, output.numpy())
			start = start + batch_size
	metrics_info = metrics_util.compute_metrics(y, y_pred, metrics)
	return (y_pred, metrics_info)
