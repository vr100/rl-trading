import matplotlib.pyplot as plt
import os

def plot(obs, pred, output_folder):
	if len(pred.shape) == 1:
		pred = pred.reshape((pred.shape[0], 1))
	res = obs - pred
	plt.scatter(pred, res)
	plot_path = os.path.join(output_folder, "pred-res.jpg")
	plt.savefig(plot_path, format="jpg", dpi=200, bbox_inches="tight")
	plt.close()
