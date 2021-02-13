from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer

class IdentityScaler():

	def fit_transform(self, data):
		return data

	def transform(self, data):
		return data

def get_scaler(scaler_type):
	if scaler_type == "robust":
		return RobustScaler()
	if scaler_type == "standard":
		return StandardScaler()
	if scaler_type == "quantile":
		return QuantileTransformer()
	if scaler_type == "power":
		return PowerTransformer()
	if scaler_type == "identity":
		return IdentityScaler()
	print("unknown scaler type {}".format(scaler_type))
	exit()

def scale_data(data, scaler=None, scaler_type="identity"):
	output_data = data.copy()
	if scaler is None:
		scaler = get_scaler(scaler_type)
		output_data = scaler.fit_transform(output_data)
	else:
		output_data = scaler.transform(output_data)
	return (output_data, scaler)
