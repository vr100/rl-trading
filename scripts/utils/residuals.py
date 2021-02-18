import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from utils import scaler
from scipy import stats

def scatter_plot(res, pred, output_folder):
	plt.scatter(pred, res)
	plot_path = os.path.join(output_folder, "pred-res.jpg")
	plt.savefig(plot_path, format="jpg", dpi=200, bbox_inches="tight")
	plt.close()

def normalize_plot(res, output_folder):
	plt.hist(res, bins=50, density=True)
	plot_path = os.path.join(output_folder, "norm-res.jpg")
	plt.savefig(plot_path, format="jpg", dpi=200, bbox_inches="tight")
	plt.close()

def qqplot(res, output_folder):
	(norm_res, _) = scaler.scale_data(res, scaler_type="standard")
	sm.qqplot(norm_res, dist=stats.norm, line ="45", fit=True)
	plot_path = os.path.join(output_folder, "qqplot-res.jpg")
	plt.savefig(plot_path, format="jpg", dpi=200, bbox_inches="tight")
	plt.close()

def plot(obs, pred, output_folder):
	if len(pred.shape) == 1:
		pred = pred.reshape((pred.shape[0], 1))
	res = obs - pred
	scatter_plot(res, pred, output_folder)
	normalize_plot(res, output_folder)
	qqplot(res, output_folder)
