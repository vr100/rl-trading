# Scripts

## autoencoder-train.py

	python3 autoencoder-train.py [-h] --data_path DATA_PATH --output_path OUTPUT_PATH [--hyperparams_path HYPERPARAMS_PATH] [--params_path PARAMS_PATH] [--fast_mode FAST_MODE]

**sample params path** : config-samples/autoencoder-params-config.json.sample

**sample hyperparams path** : config-samples/hyperparams-config.json.sample

This script is for training autoencoder model. The input data folder should contain train.csv and test.csv files

## feature-selection-borutapy.py

	python3 feature-selection-borutapy.py [-h] --data_path DATA_PATH --output_path OUTPUT_PATH --config_path CONFIG_PATH [--fast_mode FAST_MODE] [--random_mode RANDOM_MODE]

**sample config path** : config-samples/feature-selection-borutapy-config.json.sample

This script is used for selecting features using boruta algorithm. The input data folder should contain train.csv and test.csv files

## feature-selection-shap.py

	python3 feature-selection-shap.py [-h] --data_path DATA_PATH --output_path OUTPUT_PATH --config_path CONFIG_PATH [--fast_mode FAST_MODE] [--random_mode RANDOM_MODE]

**sample config path**: config-samples/feature-selection-shap-config.json.sample

This script is used for selecting features using the shap library. The input data folder should contain train.csv and test.csv files

## lstm-train.py

	python3 lstm-train.py [-h] --data_path DATA_PATH --output_path OUTPUT_PATH --regression_model_path REGRESSION_MODEL_PATH

This script is for training lstm model (after applying regression model to data). The input data folder should contain train.csv and test.csv files

## random-dataset.py

	python3 random-dataset.py [-h] --data_path DATA_PATH --output_path OUTPUT_PATH --random_count RANDOM_COUNT [--na_value {mean,zero}]

This script generates train and test csv files with specified amount of random data. The input data folder should contain train.csv and test.csv files

## regression-train.csv

	python3 regression-train.py [-h] --data_path DATA_PATH --output_path OUTPUT_PATH --config_path CONFIG_PATH --features_path FEATURES_PATH] [--autoencoder_path AUTOENCODER_PATH] [--fast_mode FAST_MODE]

**sample config path** : config-samples/regression-config.json.sample

**sample mlp config path** : config-samples/regression-config-mlp.json.sample

This script is for training regression models. The features_path if provided will be used to select the specified features from the input data. The autoencoder_path if provided will be applied before regression training. The input data folder should contain train.csv and test.csv files

## rl-hyperopt.py

	python3 rl-hyperopt.py [-h] --data_path DATA_PATH --output_path OUTPUT_PATH --hyperparams_path HYPERPARAMS_PATH --config_path  CONFIG_PATH [--features_path FEATURES_PATH]

**sample config path** : config-samples/rl-base-config.json.sample

**sample hyperparam path**: config-samples/rl-hyperparams.json.sample

This script is used for hyperparameter optimization for reinforcement learning algorithms. The features_path if provided will be used to select the specified features from the input data. The input data folder should contain train.csv and test.csv files

## rl-train.py

	python3 rl-train.py [-h] --data_path DATA_PATH --output_path OUTPUT_PATH --config_path CONFIG_PATH [--features_path FEATURES_PATH] [--fast_mode FAST_MODE] [--random_mode RANDOM_MODE]

**sample config path** : config-samples/rl-config.json.sample

This script is used for training using reinforcement learning algorithms. The features_path if provided will be used to select the specified features from the input data. The input data folder should contain train.csv and test.csv files

## split-dataset.py

	python3 split-dataset.py [-h] --data_path DATA_PATH --output_path OUTPUT_PATH [--time_sensitive TIME_SENSITIVE]

This script reads the given data csv into memory, splits into train and test data and stores them as separate csv files. The time_sensitive parameter decides if the data split is random or with respect to some time column
