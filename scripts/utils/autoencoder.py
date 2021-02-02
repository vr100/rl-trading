import torch.nn as nn
import torch.optim as optim
import math, torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import utils.metrics as metrics_util
import utils.misc as helper

BATCH_SIZE = 256
LEARNING_RATE = 0.001
DEVICE = "cpu"

class Autoencoder(nn.Module):

	def get_nearest_power_of_2(self, value):
		index = 0
		while value >= 2:
			value = value / 2
			index = index + 1
		return int(math.pow(2, index))

	def __init__(self, input_size, expected_size, dropout):
		super(Autoencoder, self).__init__()

		nearest_2_power = self.get_nearest_power_of_2(input_size)
		encode_list = [ nn.BatchNorm1d(input_size),
			nn.Linear(input_size, nearest_2_power) ]
		size = nearest_2_power
		while size > (2 * expected_size):
			new_size = size // 2
			enc = nn.Linear(in_features=size, out_features=new_size)
			encode_list.extend([nn.GELU(), nn.Dropout(dropout),
				nn.BatchNorm1d(size), enc])
			size = new_size
		if size > expected_size:
			enc = nn.Linear(in_features=size, out_features=expected_size)
			encode_list.extend([nn.GELU(), nn.Dropout(dropout),
				nn.BatchNorm1d(size), enc])
		encode_list.append(nn.GELU())
		self.encode = nn.Sequential(*encode_list)

		nearest_2_power = self.get_nearest_power_of_2(expected_size) * 2
		decode_list = [ nn.BatchNorm1d(expected_size),
			nn.Linear(in_features=expected_size, out_features=nearest_2_power) ]
		size = nearest_2_power
		while size < (input_size // 2):
			new_size = size * 2
			dec = nn.Linear(in_features=size, out_features=new_size)
			decode_list.extend([nn.GELU(), nn.Dropout(dropout),
				nn.BatchNorm1d(size), dec])
			size = new_size
		if size < input_size:
			dec = nn.Linear(in_features=size, out_features=input_size)
			decode_list.extend([nn.GELU(), nn.Dropout(dropout),
				nn.BatchNorm1d(size), dec])
		decode_list.append(nn.GELU())
		self.decode = nn.Sequential(*decode_list)

	def print_details(self):
		print("Encoder: {}".format(self.encode))
		print("Decoder: {}".format(self.decode))

	def forward(self, x):
		x = self.encode(x)
		x = self.decode(x)
		return x

def get_model(input_size, expected_size, dropout):
	return Autoencoder(input_size, expected_size, dropout)

def train(model, x, epochs=25, lr=LEARNING_RATE,
	batch_size=BATCH_SIZE):
	x = helper.get_torch_representation(x, DEVICE)
	x_with_noise = helper.add_gaussian_noise(x)
	dataset = TensorDataset(x_with_noise)
	loader = DataLoader(dataset, batch_size=batch_size)
	loss_fn = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)
	model.train()
	for i in range(epochs):
		running_loss = 0.0
		index = 0
		for data in loader:
			optimizer.zero_grad()
			output = model(data[0])
			loss = loss_fn(data[0], output)
			#loss = add_reg_loss(model, loss)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			index += 1
		epoch_loss = running_loss / index
		print ("Finished epoch {}/{}, loss: {}".format(i+1, epochs,
			epoch_loss))
	model.eval()
	return model

def evaluate(model, x, metrics, batch_size=BATCH_SIZE):
	pred = infer(model, x, batch_size)
	metrics_info = metrics_util.compute_metrics(x, pred, metrics)
	return metrics_info

def infer(model, x, batch_size=BATCH_SIZE):
	model.eval()
	x = helper.get_torch_representation(x, DEVICE)
	dataset = TensorDataset(x)
	loader = DataLoader(dataset, batch_size=batch_size)
	pred = np.empty(shape=(0, x.shape[1]))
	with torch.no_grad():
		for data in loader:
			output = model(data[0])
			pred = np.append(pred, output.numpy(), axis=0)
	return pred

def encode(model, x, output_size, batch_size=BATCH_SIZE):
	model.eval()
	x = helper.get_torch_representation(x, DEVICE)
	dataset = TensorDataset(x)
	loader = DataLoader(dataset, batch_size=batch_size)
	encoded_x = np.empty(shape=(0, output_size))
	with torch.no_grad():
		for data in loader:
			output = model.encode(data[0])
			encoded_x = np.append(encoded_x, output.numpy(),
				axis=0)
	return encoded_x
