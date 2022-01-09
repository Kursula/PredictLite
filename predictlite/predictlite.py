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

# Pytorch
import torch
import torch.nn as nn


class PredictLite:
    
    def __init__(
        self,
        input_signals: list = None, 
        input_length: int = None, 
        output_signal: str = None,
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
                
        self.input_signals = input_signals
        self.input_length = input_length
        self.output_signal = output_signal
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

        self.preproc_options = ['minmax', 'z-norm', 'none']
        self.preproc_stats = {}
        self.eps = 1e-12
        
        self.__version__ = 0.1
        

    def nn_init(self) -> None: 
        """
        Initialize the neural network model. 
        """
        layers = []

        # Input
        input_len = self.input_length * len(self.input_signals)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(input_len, input_len))
        prev_layer_neuron_n = input_len
        
        # Hidden layers
        if self.hidden_layer_n > 0:
            
            # Add neuron counts if not specified by user
            while len(self.hidden_layer_neurons) < self.hidden_layer_n:
                self.hidden_layer_neurons.append(input_len)
            
            # Add first ReLU layer
            layers.append(nn.ReLU())
            
            # Add the hidden layers
            for i in range(self.hidden_layer_n): 
                this_layer_neuron_n = self.hidden_layer_neurons[i]
                layers.append(nn.Linear(prev_layer_neuron_n, this_layer_neuron_n))
                layers.append(nn.ReLU())
                prev_layer_neuron_n = this_layer_neuron_n

        # Output
        output_len = self.output_length
        layers.append(nn.Linear(prev_layer_neuron_n, output_len))
        self.model = nn.Sequential(*layers)

        # Loss and optimizer
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            
    def nn_forward(self, x: torch.tensor) -> torch.tensor:
        """
        Training forward pass. 
        Not used in inference phase. 
        """
        y_pred = self.model(x)
        return y_pred
    
    
    def nn_loss(self, y_pred, y_true): 
        """
        Training loss function. 
        """
        return self.loss_func(y_pred, y_true)

    
    def nn_predict(self, x: torch.tensor) -> np.ndarray: 
        """
        Inference processing. 
        """
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x.unsqueeze(0)).detach()[0].numpy()
        return y_pred

    
    def nn_fit(self, train_loader, test_loader) -> None:
        """
        Neural network model training and testing. 
        """
        self.train_losses = []
        self.test_losses = []
        
        for epoch in range(self.epochs):
            # Training loop 
            self.model.train()
            epoch_losses = []
            for data, target in train_loader:
                self.optimizer.zero_grad()
                output = self.nn_forward(data)
                loss = self.nn_loss(output, target)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
                
            self.train_losses.append(np.mean(epoch_losses))

            # Test loop 
            self.model.eval()
            epoch_losses = []
            with torch.no_grad():
                for data, target in test_loader:
                    output = self.nn_forward(data)
                    loss = self.nn_loss(output, target)
                    epoch_losses.append(loss.item())
                    
            self.test_losses.append(np.mean(epoch_losses))
            self.logging('epoch: {:3}, train loss: {:0.5f}, test loss: {:0.5f}'.format(
                    epoch,
                    self.train_losses[-1],
                    self.test_losses[-1]
                )
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
        
        # Fit preprocessing 
        self.preprocessing_fit(data)
        
        # Preprocess the data
        proc_data = self.preprocess(data)        
        self.logging('Preprocessing done')
        
        # Create training samples
        split_i = int(len(proc_data) * self.train_test_split)
        train_samples = self.create_samples(
            data=proc_data.iloc[0 : split_i], 
            sample_count=self.train_sample_n
        )
        # Convert training samples to dataloader format
        train_loader = torch.utils.data.DataLoader(
            train_samples, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        self.logging('Training samples created')

        # Create testing samples
        test_samples = self.create_samples(
            data=proc_data.iloc[split_i :], 
            sample_count=self.test_sample_n
        )
        # Convert testing samples to dataloader format
        test_loader = torch.utils.data.DataLoader(
            test_samples, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        self.logging('Testing samples created')

        # Initialize neural network
        self.logging('Setting up and training the model')
        self.nn_init()

        # Fit tne neural network
        self.nn_fit(train_loader, test_loader)
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
        proc_data = self.preprocess(data.loc[ts_start : ts_end, self.input_signals])
        input_tensor = self.create_input_tensor(proc_data, ts_end)
        
        # Make prediction
        prediction = self.nn_predict(input_tensor)
        
        # Scale 
        prediction = self.postprocess(prediction)
        
        # Create dataframe for results
        period = timedelta(seconds=self.data_sample_period)
        ts_index = [ts_end + (1 + i) * period for i in range(self.output_length)]        
        pred_df = pd.DataFrame(index=ts_index, data={self.output_signal : prediction})
        return pred_df
        
    
    def preprocessing_fit(self, data: pd.DataFrame) -> None: 
        """
        Fit the preprocessing parameters. 
        """
        # Check the given parameters and setup default method if parameters are not given. 
        if self.input_preprocessing is None: 
            self.input_preprocessing = {}
            
        for col in self.input_signals: 
            if col not in self.input_preprocessing.keys():
                self.input_preprocessing[col] = 'none'
        
        for col, preproc in self.input_preprocessing.items():
            if preproc not in self.preproc_options:
                raise ValueError('Invalid preprocessing "{}" for signal "{}"'\
                                 .format(preproc, col))

        # Initialize signal stats dicts 
        self.preproc_stats['signal_min_values'] = {}
        self.preproc_stats['signal_max_values'] = {}
        self.preproc_stats['signal_mean_values'] = {}
        self.preproc_stats['signal_std_values'] = {}
        
        # Calculate stats
        for col in self.input_signals: 
            self.preproc_stats['signal_min_values'][col] = data[col].min()
            self.preproc_stats['signal_max_values'][col] = data[col].max()
            self.preproc_stats['signal_mean_values'][col] = data[col].mean()
            self.preproc_stats['signal_std_values'][col] = data[col].std()

    
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
    
    
    def postprocess(self, data: np.ndarray) -> np.ndarray: 
        """
        Post-process the prediction results, i.e. scale the values back to original scale. 
        """
        col = self.output_signal
        if self.input_preprocessing[col] == 'minmax':
            min_val = self.preproc_stats['signal_min_values'][col]
            max_val = self.preproc_stats['signal_max_values'][col]
            data = data * (max_val - min_val) + min_val
        elif self.input_preprocessing[col] == 'z-norm':
            mean_val = self.preproc_stats['signal_mean_values'][col]
            std_val = self.preproc_stats['signal_std_values'][col]
            data = data * std_val + mean_val
        return data
    
    
    def create_input_tensor(self, data: pd.DataFrame, timestamp : pd.Timestamp): 
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
    
    
    def create_target_tensor(self, data: pd.DataFrame, timestamp : pd.Timestamp): 
        """
        Covert dataframe data to a Torch tensor in model output configuration. 
        Timestamp gives the last row in input data. 
        """
        i_start = data.index.get_loc(timestamp, method='pad') + 1
        ts_start = data.index[i_start] 
        i_end = i_start + self.output_length - 1
        ts_end = data.index[i_end]
        target_values = data.loc[ts_start : ts_end, self.output_signal].values
        target_tensor = torch.from_numpy(target_values).float()                          
        return target_tensor
    
    
    def create_samples(self, data: pd.DataFrame, sample_count : int) -> list:
        """
        Create input and target sample pairs for model training. 
        """
        samples = []
        
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
            target_tensor = self.create_target_tensor(data, timestamp)
            samples.append([input_tensor, target_tensor])
        
        return samples
    
    
    def save(self, filename: str) -> None: 
        """
        Save parameters to file.  
        """
        # Model parameters 
        model_params = self.model.state_dict()
        
        # Merge object parameters to same dict
        model_params['object_params'] = self.get_params_dict()
        
        torch.save(model_params, fn + '.pth')
    
    
    def get_params_dict(self) -> dict:
        """
        Put all relevant parameters to dictionary for model saving purposes.
        """
        params = self.__dict__.copy()
        ignore_keys = ['model', 'loss_func', 'optimizer']
        for k in ignore_keys: 
            params.pop(k)
        
        return params 
        
    
    def load(self, filename: str) -> None: 
        """
        Load parameters from file.
        """
                
        model_params = torch.load(fn)
        object_params = model_params.pop('object_params')
                
        # Populate object parameters from dictionary
        for k, v in object_params.items():
            command = 'self.{} = v'.format(k)
            exec(command)

        # Setup model from loaded parameters
        self.nn_init()
        self.model.load_state_dict(model_params)

    
    def logging(self, txt: str) -> None: 
        """
        Logging and printing. 
        """
        if self.verbose:
            print(txt)
        