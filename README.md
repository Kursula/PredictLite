# PredictLite

PredictLite is a lightweight timeseries data prediction framework that builds on top of popular PyTorch and Pandas packages. 

## Features

PredictLite can perform linear regression and nonlinear (deep neural network) regression using one or more input signals to predict one or more output signal. Optinal features include e.g. datetime embeddings that can be used to train time dependencies, such as weekly patterns and seasonality. Similarly categorical embeddings can be used to incorporate categorical information to the prediction modeling. 

PredictLite has basic capabilities for NaN/Null value imputation and data pre- and post-processing.  

## Installing

Copy the predictlite folder contents to your project and integrate it by following the example code. 

## Data formats

Pandas DataFrame format is supported. The DataFrame must have timestamp index. Input signals must be organized into columns. Predictions are provided in a similar DataFrame format.

Only lagged inputs are supported, i.e. the model will not utilize known future variables. Model will use input data from the given timestamp and the input_length - 1 earlier values. 

Categorical inputs can be in any format, e.g. string, bool, int, etc. Categorical data is internally enumerated and mapped to embedding vectors. Out of distribution (OOD) embedding is trained and used during inference for categorical values that are not present in training data. For categorical data the model uses the values from the last timestamp in the input data. 

Datetime embeddings are internally calculated from the data timestamps. Also the datetime embeddings include OOD embedding that is used for values not present in the training data. Such situation can occur e.g. when using year embeddings and making predictions for future years where there is no training data available. For datetime embeddigs the model uses the last timestamp in the input data. 


## Model setup 

The PredictLite class initialization takes the following input parameters.

Mandatory parameters when setting up a new model:
- input_signals: list of column names that specify the model input signals. 
- input_length: number of time steps to be used in model input. Same length is applied to all inputs.  
- output_signals: list of columns that are to be predicted. These must be also in input signals. 
- output_length: number of time steps to be predicted. Uses same sample period as the inputs. 
- data_sample_period: specifies the data sample period in seconds (float). This is used to generate the prediction timestamps.

Mandatory parameters when using a saved model: 
- load_from_file: string path to saved model. All necessary parameters will be fetched from the file.

Optional parameters when setting up new model:
- hidden_layer_n: number of hidden layers to be used in the model. Default = 0.
- hidden_layer_neurons: list containign the hidden layer neuron count. List is ordered from model input towards the output. Default configuration uses hidden layers that are same size as input layer. 
- input_preprocessing: dictionary containing input preprocessing (scaling and normalization) configuration. The dictinary must have format {'column_name_1' : 'preprocessing_1', 'column_name_2' : 'preprocessing_2', ...} where preprocessing can be one of 'minmax', 'z-norm' or 'none'. Minmax scales the signal values to [0, 1] range. Z-norm scales the signal mean to 0 and standard deviation to 1. None does nothing. Default value is none. 
- datetime_embeddings: list of datetime aggregates that are to be used as embeddings in the model. Using these embeddings improves seasonality and periodic pattern modeling. For example, month embeddings will learn annual seasonality patterns if those exist in the data. Possible values are year, month, day of week, hour and minute. More than one can be used. Default is none. All embedding calculation will be done automatically using the data timestamp as datetime source. Embeddings are calculated for the last (most recent) timestamp value in the model input. 
- datetime_embedding_dim: specifies length of the datetime embedding vectors. Default is 20. Same setting applies to all datetime embeddings. 
- categorical_inputs: list of categorical data column names. These will be used to generate categorical embeddings. 
- categorical_embedding_dim: specifies length of the categorical embedding vectors. Default is 20. Same setting applies to all categorical embeddings. 
- embedding_ood_ratio: ratio of training samples to be used to train out of distribution (OOD) embeddings. Default value = 0.02 meaning that 2% of training samples will use OOD categorical values. Does not apply to test data. 
- interpolate_nan: enable/disable NaN value interpolation. Default value is True. Will perform linear interpolation using previous and next valid values. Will not be able to process NaN values at either end of the input DataFrame. 
- zerofill_nan: enable/disable NaN value zero filling. Default value is True. If also interpolate_nan is True, the interpolation is performed first and the remaining NaN values are zerofilled. 
- train_test_split: float between 0 and 1. Gives the ratio of data to be used in model training. Default is 0.8, which means that 80% of data rows are used in training and the last 20% are used in testing.
- train_sample_n: integer value that specifies the number of trainig samples. Default is None and it causes all data rows to be used in model training.
- test_sample_n: integer value that specifies the number of test samples. Default is None and it causes all data rows to be used in model testing.
- learning_rate: neural network learning rate. Default = 1e-4.
- batch_size: neural network training batch size. Default = 64.
- epochs: neural network training epoch count. Default = 30.
- random_seed: integer seed for random function. Set this when results must be possible to reproduce. Default = None. 
- verbose: boolean value that enables/disables status and debug prints. Default = True.


## Inference 

Inference is run using the PredictLite.predict method. It takes two parameters as input
- dataset in Pandas DataFrame format (same requirements as in model training case). 
- timestamp for the last datapoint to be used in model input. If not given, the prediction will be based on the data at the end of the dataset. 

Returns a DataFrame containing the predictions. 

## Model save and load
PredictLite saves all configuration parameters and neural network parameters to single file. 

Save the trained model by running save method. Takes filename as parameter. 