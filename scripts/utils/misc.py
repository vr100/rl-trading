import numpy as np
import torch

def add_gaussian_noise_for_numpy(data, mean=0, std=1):
	noise = std * np.random.randn(data.size) + mean
	noise = np.reshape(noise, data.shape)
	return data + noise

def add_gaussian_noise_for_tensor(data, mean=0, std=1):
	return data + torch.randn(data.size()) * std + mean

def add_gaussian_noise(data, mean=0, std=1):
	if isinstance(data, np.ndarray):
		return add_gaussian_noise_for_numpy(data, mean, std)
	return add_gaussian_noise_for_tensor(data, mean, std)

def get_torch_representation(data, device):
	if isinstance(data, np.ndarray):
		numpy_data = data
	else:
		numpy_data = data.to_numpy()
	return torch.from_numpy(numpy_data).to(device).float()
