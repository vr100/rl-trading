from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer

def get_scaler(scaler_type):
	if scaler_type == "robust":
		return RobustScaler()
	if scaler_type == "standard":
		return StandardScaler()
	if scaler_type == "quantile":
		return QuantileTransformer()
	if scaler_type == "power":
		return PowerTransformer()
	print("unknown scaler type {}".format(scaler_type))
	exit()

def scale_data(data, scaler=None, scaler_type="standard"):
	output_data = data.copy()
	if scaler is None:
		scaler = get_scaler(scaler_type)
		output_data = scaler.fit_transform(output_data)
	else:
		output_data = scaler.transform(output_data)
	return (output_data, scaler)
