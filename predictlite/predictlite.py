################################################################################
# This is PredictLite, a lightweight time series prediction model using        
# PyTorch and basic Numpy and Pandas processing functions.
#
# Copyright Mikko Kursula 2022. MIT license. 
################################################################################

# General
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Callable

# Pytorch
import torch
import torch.nn as nn

        
class PredictionModel(nn.Module):

    def __init__(self, 
                 input_length, 
                 input_signals,
                 hidden_layer_neurons,
                 hidden_layer_n, 
                 output_length,
                 output_signals
                ):
        super().__init__()
        
        self.output_length = output_length
        self.output_signals = output_signals
        
        layers = []

        # Input
        input_len = input_length * len(input_signals)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(input_len, input_len))
        prev_layer_neuron_n = input_len
        
        # Hidden layers
        if hidden_layer_n > 0:
            
            # Add first ReLU layer
            layers.append(nn.ReLU())

            # Add the hidden layers
            for i in range(hidden_layer_n): 
                this_layer_neuron_n = hidden_layer_neurons[i]
                layers.append(nn.Linear(prev_layer_neuron_n, this_layer_neuron_n))
                layers.append(nn.ReLU())
                prev_layer_neuron_n = this_layer_neuron_n

        # Output
        output_len = output_length * len(output_signals)
        layers.append(nn.Linear(prev_layer_neuron_n, output_len))
        self.model = nn.Sequential(*layers)
            
            
    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Training forward pass. 
        Not used in inference phase. 
        """
        y_pred = self.model(x)
        return y_pred
    
    
    def loss(self, y_pred, y_true): 
        """
        Training loss function. 
        """
        loss = nn.MSELoss()
        return loss(y_pred, y_true)

    
    def predict(self, x: torch.tensor) -> torch.tensor: 
        """
        Inference processing. 
        """
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x.unsqueeze(0)).detach()[0]
        return y_pred


    def save(self, filename: str) -> None: 
        fn = filename.split('.')[0] + '.model'
        torch.save(self.state_dict(), fn)
            
    
    def load(self, filename: str) -> None: 
        fn = filename.split('.')[0] + '.model'
        self.load_state_dict(torch.load(fn))

        
class PredictionDataset(torch.utils.data.Dataset):
    
    def __init__(self, float_inputs, targets):
        self.float_inputs = float_inputs
        self.targets = targets
            
            
    def __len__(self):
        return len(self.targets)
    
    
    def __getitem__(self, idx):
        return self.float_inputs[idx], self.targets[idx]

    
class PredictionModelTrainer:
    
    def __init__(self, 
                 model: PredictionModel, 
                 learning_rate: float, 
                 epochs: int, 
                 logging: Callable
                ): 
        
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.logging = logging
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    
    def fit(self, train_loader, test_loader) -> None:
        """
        Neural network model training and testing. 
        """
        train_losses = []
        test_losses = []
        
        for epoch in range(self.epochs):
            # Training loop 
            self.model.train()
            epoch_losses = []
            for data, target in train_loader:
                self.optimizer.zero_grad()
                output = self.model.forward(data)
                loss = self.model.loss(output, target)     
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
                
            train_losses.append(np.mean(epoch_losses))

            # Test loop 
            self.model.eval()
            epoch_losses = []
            with torch.no_grad():
                for data, target in test_loader:
                    output = self.model.forward(data)
                    loss = self.model.loss(output, target)
                    epoch_losses.append(loss.item())
                    
            test_losses.append(np.mean(epoch_losses))
            self.logging('epoch: {:3}, train loss: {:0.5f}, test loss: {:0.5f}'.format(
                    epoch,
                    train_losses[-1],
                    test_losses[-1]
                )
            )
        return train_losses, test_losses


class PredictionModelUtils:

    def __init__(self, 
                 input_length: int, 
                 input_signals: list,
                 output_length: int, 
                 output_signals: list,
                ):
        self.input_length = input_length
        self.input_signals = input_signals
        self.output_length = output_length 
        self.output_signals = output_signals

    
    def create_input_tensor(self, data: pd.DataFrame, timestamp : pd.Timestamp) -> torch.tensor: 
        """
        Covert dataframe data to a Torch tensor in model input configuration. 
        Timestamp gives the last row in input data. 
        """
        i_end = data.index.get_loc(timestamp, method='pad')
        i_start = i_end - self.input_length + 1
        ts_start = data.index[i_start]
        input_values = data.loc[ts_start : timestamp, self.input_signals].values
        input_tensor = torch.from_numpy(input_values).float().permute(1, 0)                           
        return input_tensor
    
    
    def create_target_tensor(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> torch.tensor: 
        """
        Covert dataframe data to a Torch tensor in model output configuration. 
        Timestamp gives the last row in input data. 
        """
        i_start = data.index.get_loc(timestamp, method='pad') + 1
        ts_start = data.index[i_start] 
        i_end = i_start + self.output_length - 1
        ts_end = data.index[i_end]
        target_values = data.loc[ts_start : ts_end, self.output_signals].values.flatten(order='F')
        target_tensor = torch.from_numpy(target_values).float()                          
        return target_tensor
    
    
    def parse_prediction_from_tensor(self, output_tensor: torch.tensor) -> pd.DataFrame:
        values = output_tensor.numpy().reshape((self.output_length, len(self.output_signals)), order='F')
        pred = pd.DataFrame(data=values, columns=self.output_signals)
        return pred

    
    def create_samples(self, data: pd.DataFrame, sample_count : int) -> list:
        """
        Create input and target sample pairs for model training. 
        """
        inputs = []
        targets = []
        
        # Determine the number of samples to be created and index values to be used. 
        possible_timestamps = data.index[self.input_length : -self.output_length]
        
        if sample_count is None: 
            # All data is used in training
            sample_timestamps = possible_timestamps
        else: 
            # Random samples are selected for training
            if len(possible_timestamps) < sample_count: 
                raise ValueError('Number of samples {} exceeds the amount of data {}.'
                                 .format(sample_count, len(possible_timestamps)))
            sample_timestamps = np.random.choice(possible_timestamps, sample_count, replace=False)
            
        # Crete samples
        for timestamp in sample_timestamps:
            input_tensor = self.create_input_tensor(data, timestamp)
            inputs.append(input_tensor)
            target_tensor = self.create_target_tensor(data, timestamp)
            targets.append(target_tensor)
        
        return inputs, targets

    
    def create_data_loader(self, dataset: PredictionDataset, batch_size, shuffle): 
        # Convert training samples to dataloader format
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle
        )
        return data_loader


class DataPreAndPostProcessor:

    proc_options = ['minmax', 'z-norm', 'none']
    
    def __init__(self,
                 input_signals: list,
                 resample_data: bool, 
                 data_sample_period: float,
                 interpolate_nan: bool,
                 zerofill_nan: bool,
                 input_preprocessing: dict, 
                 output_signals: list,
                ):

        self.input_signals = input_signals
        self.resample_data = resample_data
        self.data_sample_period = data_sample_period
        self.interpolate_nan = interpolate_nan
        self.zerofill_nan = zerofill_nan
        self.input_preprocessing = input_preprocessing
        self.output_signals = output_signals
        
        self.preproc_stats = {}
        self.eps = 1e-12
        
        
    def fit(self, data: pd.DataFrame) -> None: 
        """
        Fit the preprocessing parameters. 
        """

        # Initialize signal stats dicts 
        self.preproc_stats['signal_min_values'] = {}
        self.preproc_stats['signal_max_values'] = {}
        self.preproc_stats['signal_mean_values'] = {}
        self.preproc_stats['signal_std_values'] = {}
        
        # Calculate stats
        for col in self.input_signals: 
            self.preproc_stats['signal_min_values'][col] = data[col].dropna().min()
            self.preproc_stats['signal_max_values'][col] = data[col].dropna().max()
            self.preproc_stats['signal_mean_values'][col] = data[col].dropna().mean()
            self.preproc_stats['signal_std_values'][col] = data[col].dropna().std()

    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame: 
        """
        Preprocess the data.
        """
        proc_data = data.copy()
        
        # Resample
        if self.resample_data: 
            proc_data = proc_data.resample('{}s'.format(self.data_sample_period)).mean()
        
        # Process NaN values
        if self.interpolate_nan: 
            proc_data = proc_data.interpolate()
        if self.zerofill_nan:
            proc_data = proc_data.fillna(0)
        
        # Scale values
        for col in self.input_signals: 
            if self.input_preprocessing[col] == 'minmax':
                min_val = self.preproc_stats['signal_min_values'][col]
                max_val = self.preproc_stats['signal_max_values'][col]
                proc_data[col] = (proc_data[col] - min_val) / (max_val - min_val)
            
            elif self.input_preprocessing[col] == 'z-norm':
                mean_val = self.preproc_stats['signal_mean_values'][col]
                std_val = self.preproc_stats['signal_std_values'][col]
                proc_data[col] = (proc_data[col] - mean_val) / (std_val + self.eps)

        return proc_data
    
    
    def postprocess(self, data: pd.DataFrame) -> pd.DataFrame: 
        """
        Post-process the prediction results, i.e. scale the values back to original scale. 
        Outputs share the same preprocessing with inputs. 
        """
        proc_data = data.copy()
        
        for col in self.output_signals:
            if self.input_preprocessing[col] == 'minmax':
                min_val = self.preproc_stats['signal_min_values'][col]
                max_val = self.preproc_stats['signal_max_values'][col]
                proc_data[col] = proc_data[col] * (max_val - min_val) + min_val
            elif self.input_preprocessing[col] == 'z-norm':
                mean_val = self.preproc_stats['signal_mean_values'][col]
                std_val = self.preproc_stats['signal_std_values'][col]
                proc_data[col] = proc_data[col] * std_val + mean_val

        return proc_data

    
    def get_params(self) -> dict: 
        return self.preproc_stats
    
    
    def set_params(self, params: dict) -> None: 
        self.preproc_stats = params
    

class PredictLite:
    
    def __init__(
        self,
        load_from_file: str = None,
        input_signals: list = None, 
        input_length: int = None, 
        output_signals: list = None,
        output_length: int = None, 
        resample_data: bool = False,
        data_sample_period: float = None, 
        hidden_layer_n: int = 0,
        hidden_layer_neurons: list = [],
        input_preprocessing: dict = None,
        zerofill_nan: bool = True,
        interpolate_nan: bool = True,
        train_sample_n: int = None,
        test_sample_n: int = None,
        train_test_split: float = 0.8, 
        learning_rate: float = 1e-4, 
        batch_size: int = 64,
        epochs: int = 30,
        random_seed: int = None, 
        verbose: bool = True
    ): 

        self.__version__ = 0.3

        self.load_from_file = load_from_file
        self.input_signals = input_signals
        self.input_length = input_length
        self.output_signals = output_signals
        self.output_length = output_length 
        self.resample_data = resample_data
        self.data_sample_period = data_sample_period
        self.hidden_layer_n = hidden_layer_n
        self.hidden_layer_neurons = hidden_layer_neurons
        self.input_preprocessing = input_preprocessing
        self.zerofill_nan = zerofill_nan
        self.interpolate_nan = interpolate_nan
        self.train_sample_n = train_sample_n
        self.test_sample_n = test_sample_n
        self.train_test_split = train_test_split
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_seed = random_seed
        self.verbose = verbose

        if self.load_from_file is not None:
            self.load(self.load_from_file)
            self.check_and_parse_args()
        else: 
            self.check_and_parse_args()
            self.pipeline_init()
        
        
    def check_and_parse_args(self) -> None: 
        
        if (self.load_from_file is None) and (self.input_signals is None): 
            raise ValueError('Mandatory arguments or load_from_file not specified.')
        
        # Check inputs and outputs
        if len(self.input_signals) == 0: 
            raise ValueError('No input signals specified')
        if len(self.output_signals) == 0: 
            raise ValueError('No output signals specified')
            
        # Check that all outputs are also in inputs. 
        for col in self.output_signals:
            if col not in self.input_signals: 
                raise ValueError('Signal {} not specified as input.'.format(col))        
        
        # Check the given parameters and setup default method if parameters are not given. 
        if self.input_preprocessing is None: 
            self.input_preprocessing = {}
        for col in self.input_signals: 
            if col not in self.input_preprocessing.keys():
                self.input_preprocessing[col] = 'none'

        # Check that given preprocessing is available
        for col, preproc in self.input_preprocessing.items():
            if preproc not in DataPreAndPostProcessor.proc_options:
                raise ValueError('Invalid preprocessing "{}" for signal "{}"'.format(preproc, col))
                
        # Model parameter parsing. Add default neuron counts if not specified by user.
        while len(self.hidden_layer_neurons) < self.hidden_layer_n:
            default_neuron_n = self.input_length * len(self.input_signals)
            self.hidden_layer_neurons.append(default_neuron_n)
        
    
    def pipeline_init(self) -> None: 
        
        self.model = PredictionModel(
            self.input_length, 
            self.input_signals,
            self.hidden_layer_neurons,
            self.hidden_layer_n, 
            self.output_length,
            self.output_signals
        )
        
        self.model_trainer = PredictionModelTrainer(
            self.model, 
            self.learning_rate, 
            self.epochs, 
            self.logging
        )
        
        self.model_utils = PredictionModelUtils(
            self.input_length,
            self.input_signals,
            self.output_length,
            self.output_signals
        )
        
        self.data_preproc = DataPreAndPostProcessor(
             self.input_signals,
             self.resample_data, 
             self.data_sample_period,
             self.interpolate_nan,
             self.zerofill_nan,
             self.input_preprocessing, 
             self.output_signals
        )
 

    def model_summary(self) -> None: 
        print(self.model)
        

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the entire package, i.e. preprocessing, neural network, 
        postprocessing and other possible parameters. 
        """
        
        # Reproducibility features
        if self.random_seed is not None: 
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
        
        # Initialize neural network
        self.logging('Setting up preprocessing')
        
        # Fit preprocessing 
        self.data_preproc.fit(data)
        
        # Preprocess the data
        proc_data = self.data_preproc.preprocess(data)        
        self.logging('Building dataset')
        
        # Create training dataset
        split_i = int(len(proc_data) * self.train_test_split)
        train_inputs, train_targets = self.model_utils.create_samples(
            data=proc_data.iloc[0 : split_i], 
            sample_count=self.train_sample_n
        )
        train_dataset = PredictionDataset(train_inputs, train_targets)
    
        # Convert training samples to dataloader format
        train_loader = self.model_utils.create_data_loader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )

        # Create testing dataset
        test_inputs, test_targets = self.model_utils.create_samples(
            data=proc_data.iloc[split_i :], 
            sample_count=self.test_sample_n
        )
        test_dataset = PredictionDataset(test_inputs, test_targets)

        # Convert testing samples to dataloader format
        test_loader = self.model_utils.create_data_loader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )

        # Fit the neural network
        self.logging('Training the model')
        self.train_losses, self.test_losses = self.model_trainer.fit(train_loader, test_loader)
        self.logging('Model training done')
        
        
    def predict(self, data: pd.DataFrame, timestamp: pd.Timestamp = None) -> pd.DataFrame:
        """
        Inference prediction. 
        """
        # If prediction timestamp is not given, use the last timestamp in the data.
        if timestamp is None: 
            timestamp = data.index[-1]       
            
        # Get a slice of the data to avoid processing large amount of unnecessary rows
        i_end = data.index.get_loc(timestamp, method='pad')
        ts_end = data.index[i_end]
        est_ts_start = timestamp - timedelta(seconds=self.data_sample_period) * self.input_length
        i_start = data.index.get_loc(est_ts_start, method='pad') - 1
        ts_start = data.index[i_start]
        
        # Preprocess the data
        proc_data = self.data_preproc.preprocess(data.loc[ts_start : ts_end, self.input_signals])
        input_tensor = self.model_utils.create_input_tensor(proc_data, ts_end)
        
        # Make prediction and process it to original scale
        prediction = self.model.predict(input_tensor)
        prediction = self.model_utils.parse_prediction_from_tensor(prediction)
        prediction = self.data_preproc.postprocess(prediction)
        
        # Add timestamps to prediction
        period = timedelta(seconds=self.data_sample_period)
        ts_index = pd.Series([ts_end + (1 + i) * period for i in range(self.output_length)])       
        prediction = prediction.set_index(ts_index)
        return prediction

    
    def get_params(self) -> dict:
        """
        Put all relevant parameters to dictionary for model saving purposes.
        """
        params = {} 
        
        for k, v in self.__dict__.items():
            # Ignore class instances etc that are not necessary to save. 
            if type(v) in [str, list, dict, float, int, bool]:
                params[k] = v
        
        return params 


    def save(self, filename: str) -> None: 
        """
        Save parameters to file.  
        """
        params = {}
        params['instance_params'] = self.get_params()
        params['preproc_params'] = self.data_preproc.get_params()
        
        with open(filename, 'w') as fp:
            json.dump(params, fp)

        self.model.save(filename)
            
    
    def load(self, filename: str) -> None: 
        """
        Load parameters from file.
        """
        with open(filename, 'r') as fp:
            params = json.load(fp)
                
        # Populate object parameters from dictionary
        for k, v in params['instance_params'].items():
            command = 'self.{} = v'.format(k)
            exec(command)

        # Setup model and preprocessing from loaded parameters
        self.pipeline_init()
        self.model.load(filename)
        self.data_preproc.set_params(params['preproc_params'])

        
    def logging(self, txt: str) -> None: 
        """
        Logging and printing. 
        """
        if self.verbose:
            print(txt)
        