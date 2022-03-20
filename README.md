# PredictLite

PredictLite is a lightweight timeseries data prediction framework that builds on top of popular PyTorch and Pandas packages. 

## Features

PredictLite can perform linear regression and nonlinear (deep neural network) regression using one or more input signals to predict one or more output signal. Linear regression is implemented as two layer neural network without any nonlinear activations or hidden layers. 

It has basic capabilities for data resampling, NaN/Null value imputation and data preprocessing. All such features are built in to minimize the application development effort. 

## Installing

Copy the file predictlite.py into your project and integrate it following the example code. 

## Data formats

Pandas DataFrame format is supported. The DataFrame must have timestamp index. Input signals must be organized into columns. Predictions are provided in a similar DataFrame format.

## Model setup 

The PredictLite class initialization takes the following input parameters.

Mandatory parameters when setting up a new model:
- input_signals: list of column names that specify the model input signals. 
- input_length: number of time steps to be used in model input. Same length is applied to all inputs.  
- output_signals: list of columns that are to be predicted. These must be also in input signals. 
- output_length: number of time steps to be predicted. Uses same sample period as the inputs. 
- data_sample_period: specifies the data sample period in seconds (float). This is used to generate the prediction timestamps. Also used in data resampling.

Mandatory parameters when using a saved model: 
- load_from_file: string path to saved model. All necessary parameters will be fethecd from the file.

Optional parameters when setting up new model:
- resample_data: enable/disable data resampling. Set True if the data can contain missing timestamps or if the timestamps shall be rounded to a specific period.
- hidden_layer_n: number of hidden layers to be used in the model. Default = 0.
- hidden_layer_neurons: list containign the hidden layer neuron count. List is ordered from model input towards the output. Default configuration uses hidden layers that are same size as input layer. 
- input_preprocessing: dictionary containing input preprocessing (scaling and normalization) configuration. The dictinary must have format {'column_name_1' : 'preprocessing_1', 'column_name_2' : 'preprocessing_2', ...} where preprocessing can be one of 'minmax', 'z-norm' or 'none'. Minmax scales the signal values to [0, 1] range. Z-norm scales the signal mean to 0 and standard deviation to 1. None will not do anything to the signal. Default value is none. 
- datetime_embeddings: list of datetime aggregates that are to be used as embeddings in the model. Using these embeddings improves seasonality and periodic pattern modeling. For example, month embeddings will leanr annual weather seasonality patterns. Possible values are year, month, day of week, hour and minute. More than one can be used. Default is none. All embedding calculation will be done automatically using the data timestamp as datetime source. Embeddings are calculated for the last (most recent) timestamp value in the model input. 
- datetime_embedding_dim: specifies length of the datetime embedding vectors. Default is 20. Same setting applies to all datetime embeddings. 
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