import torch.nn as nn
import torch.optim as optim
import math, torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import utils.metrics as metrics_util
import utils.misc as helper

DEVICE = "cpu"

class MLP(nn.Module):

	def get_higher_power_of_2(self, value):
		index = 0
		temp = value * 2
		while temp >= 2:
			temp = temp / 2
			index = index + 1
		higher_power = int(math.pow(2, index))
		if (higher_power - value) < value / 2:
			higher_power = higher_power * 2
		return higher_power

	def activation_layer(self):
		return nn.LeakyReLU()

	def __init__(self, input_size, expected_size,
		config):
		super(MLP, self).__init__()
		layers = config["layers"]
		dropout = config["dropout"]

		size = self.get_higher_power_of_2(input_size)
		layer_list = [ nn.Linear(input_size, size) ]
		for i in range(layers):
			enc = nn.Linear(in_features=size, out_features=size)
			layer_list.extend([self.activation_layer(),
				nn.Dropout(dropout),
				nn.BatchNorm1d(size), enc])
		enc = nn.Linear(in_features=size, out_features=input_size)
		layer_list.extend([self.activation_layer(),
				nn.Dropout(dropout),
				nn.BatchNorm1d(size), enc])
		enc = nn.Linear(in_features=input_size, out_features=expected_size)
		layer_list.extend([self.activation_layer(),
				nn.Dropout(dropout),
				nn.BatchNorm1d(input_size), enc])
		self.layers = nn.Sequential(*layer_list)
		self.print_details()

	def print_details(self):
		print("Layers: {}".format(self.layers))

	def forward(self, x):
		x = self.layers(x)
		return x

def get_model(input_size, expected_size, config):
	return MLP(input_size, expected_size, config)

def train(model, x, y, config):
	lr = config["lr"]
	epochs = config["epochs"]
	batch_size = config["batch_size"]
	weight_decay = config["weight_decay"]
	x = helper.get_torch_representation(x,DEVICE)
	y = helper.get_torch_representation(y, DEVICE)
	if len(y.shape) == 1:
		y = torch.reshape(y, shape=(y.shape[0], 1))
	x_with_noise = helper.add_gaussian_noise(x)
	dataset = TensorDataset(x_with_noise, y)
	loader = DataLoader(dataset, batch_size=batch_size)
	loss_fn = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=lr,
		weight_decay=weight_decay)
	model.train()
	for i in range(epochs):
		running_loss = 0.0
		index = 0
		for data in loader:
			optimizer.zero_grad()
			output = model(data[0])
			loss = loss_fn(data[1], output)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			index += 1
		epoch_loss = running_loss / index
		print ("Finished epoch {}/{}, loss: {}".format(i+1, epochs,
			epoch_loss))
	model.eval()
	return model

def evaluate(model, x, y, metrics, config):
	x = helper.get_torch_representation(x, DEVICE)
	y = helper.get_torch_representation(y, DEVICE)
	if len(y.shape) == 1:
		y = torch.reshape(y, shape=(y.shape[0], 1))
	pred = infer(model, x, y.shape[1], config)
	metrics_info = metrics_util.compute_metrics(y, pred, metrics,
		multioutput="raw_values")
	return (pred, metrics_info)

def infer(model, x, output_size, config):
	batch_size = config["batch_size"]
	model.eval()
	x = helper.get_torch_representation(x, DEVICE)
	dataset = TensorDataset(x)
	loader = DataLoader(dataset, batch_size=batch_size)
	pred = np.empty(shape=(0, output_size))
	with torch.no_grad():
		for data in loader:
			output = model(data[0])
			pred = np.append(pred, output.numpy(), axis=0)
	return pred
