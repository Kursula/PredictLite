# PredictLite

PredictLite is a lightweight timeseries data prediction framework that builds on top of popular PyTorch and Pandas packages. 

## Features

The prediction model is a Multilayer Perceptron (MLP) neural network that borrows ideas from many prediction models, such as TSMixer and variety of MLP concepts. The model is constructed from three main building blocks: 
* Sequential part that processes all input features one time step at a time.
* Longitudinal part that processes the input time series one feature at a time.
* Flattened part that takes the flattened outputs from sequential and longitudinal parts and makes the final predictions.

The model trainer supports curriculum training that is usable in multi-step prediction. In curriculum training the training process starts with small number of prediction steps and gradually increases the number of prediction time steps. The curriculum training feature is implemented using loss function masking. 

The prediction can be a multivariate and multi-step. Configurable percentiles will be generated to e.g. estimate prediction confidence interval. 

The model has smoothing feature for minimizing noise in multi-step prediction. The amount of smoothing can be configured using the smoothing parameter. 

Embeddings are available for categorical and datetime features. 
PredictLite has basic capabilities for NaN/Null value imputation and data pre- and post-processing.

Model save to file and load from file. 

## Data formats

Pandas DataFrame format is supported. The DataFrame must have timestamp index. Input signals must be organized into columns. Predictions are provided in a similar DataFrame format.

Only lagged inputs are supported, i.e. the model will not utilize known future variables. Model will use input data from the given timestamp and the input_length - 1 earlier values. 

Consistent data sampling and no missing timestamps is assumed.

Categorical inputs can be in any format, e.g. string, bool, int, etc. Categorical data is internally enumerated and mapped to embedding vectors. Out of distribution (OOD) embedding is trained and used during inference for categorical values that are not present in training data.

Datetime embeddings are internally calculated from the data timestamps. Also the datetime embeddings include OOD embedding that is used for values not present in the training data. Such situation can occur e.g. when using year embeddings and making predictions for future years where there is no training data available. 

Multiple datasets are supported. The datasets (dataframes) can be e.g. time series data from different sources or segments of a time series. Train and test samples are processed from each dataset. 