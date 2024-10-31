################################################################################
# This is PredictLite, a lightweight time series prediction model using        
# PyTorch and basic Numpy and Pandas processing functions.
#
# Copyright Mikko Kursula 2022 - 2024. MIT license. 
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

# PredictLite modules 
from predictlite.model import PredictionModel
from predictlite.model_trainer import PredictionModelTrainer
from predictlite.utils import PredictionDataset, PredictionModelUtils
from predictlite.data_processor import DataPreAndPostProcessor


class PredictLite:
    
    def __init__(
        self,
        load_from_file: str = None,
        input_signals: list = None,
        input_length: int = None,
        categorical_inputs: list = [],
        output_signals: list = None,
        output_length: int = None,
        data_sample_period: float = None,
        sequential_layer_neurons: list = None,
        longitudinal_layer_neurons: list = None,
        flattened_layer_neurons: list = None,
        input_preprocessing: dict = None,
        datetime_embeddings: list = [],
        datetime_embedding_dim: int = 2,
        categorical_embedding_dim: int = 2,
        embedding_ood_ratio: float = 0.02, 
        zerofill_nan: bool = True,
        interpolate_nan: bool = True,
        percentiles: list = [0.25, 0.75],
        smoothing_weight: float = 0.5,
    ): 

        self.load_from_file = load_from_file
        self.input_signals = input_signals
        self.input_length = input_length
        self.categorical_inputs = categorical_inputs
        self.output_signals = output_signals
        self.output_length = output_length 
        self.data_sample_period = data_sample_period
        self.seq_layer_neurons = sequential_layer_neurons
        self.long_layer_neurons = longitudinal_layer_neurons
        self.flat_layer_neurons = flattened_layer_neurons
        self.input_preprocessing = input_preprocessing
        self.datetime_embeddings = datetime_embeddings
        self.datetime_embedding_dim = datetime_embedding_dim
        self.categorical_embedding_dim = categorical_embedding_dim
        self.embedding_ood_ratio = embedding_ood_ratio
        self.zerofill_nan = zerofill_nan
        self.interpolate_nan = interpolate_nan
        self.percentiles = percentiles
        self.smoothing_weight = smoothing_weight

        self.fit_done = False
        
        if self.load_from_file is not None:
            self.load(self.load_from_file)
            
        self.check_and_parse_args()
        
        
    def check_and_parse_args(self) -> None: 
        
        if (self.load_from_file is None) and (self.input_signals is None): 
            raise ValueError('Mandatory arguments or load_from_file not specified.')
        
        # Check mandatory args
        if self.input_signals is None: 
            raise ValueError('No input signals specified')
        if self.output_signals is None: 
            raise ValueError('No output signals specified')
        if self.input_length is None: 
            raise ValueError('No input length specified')
        if self.output_length is None: 
            raise ValueError('No output length specified')
        if self.data_sample_period is None: 
            raise ValueError('No data sample period specified')
        if self.datetime_embedding_dim <= 0: 
            raise ValueError('Datetime embedding dim cannot be zero or negative')
        if self.categorical_embedding_dim <= 0: 
            raise ValueError('Categorical embedding dim cannot be zero or negative')
        if self.seq_layer_neurons is None:
            raise ValueError('sequential_layer_neurons must be defined.')
        if self.long_layer_neurons is None:
            raise ValueError('longitudinal_layer_neurons must be defined.')
        if self.flat_layer_neurons is None:
            raise ValueError('flattened_layer_neurons must be defined.')
            
        # Check that all outputs are also in inputs. 
        for col in self.output_signals:
            if col not in self.input_signals: 
                raise ValueError('Signal {} not specified as input.'.format(col))        
        
        # Check the given preprocessing parameters and setup default method if parameters are not given. 
        if self.input_preprocessing is None: 
            self.input_preprocessing = {}
        for col in self.input_signals: 
            if col not in self.input_preprocessing.keys():
                self.input_preprocessing[col] = 'none'

        # Check that given preprocessing is available
        for col, preproc in self.input_preprocessing.items():
            if preproc not in DataPreAndPostProcessor.proc_options:
                raise ValueError('Invalid preprocessing "{}" for signal "{}"'.format(preproc, col))
                
        # Check that given datetime embeddings are available
        for emb in self.datetime_embeddings:
            if emb not in DataPreAndPostProcessor.datetime_emb_options:
                raise ValueError('Invalid datetime embedding {}'.format(emb))

        # Check embedding dimensions 
        if self.datetime_embedding_dim < 2:
            raise ValueError('datetime_embedding_dim must be minimum 2')    
        if self.categorical_embedding_dim < 2:
            raise ValueError('categorical_embedding_dim must be minimum 2')         

            
    def setup_model_utils(self) -> None: 
        self.model_utils = PredictionModelUtils(
            input_length=self.input_length,
            input_signals=self.input_signals,
            output_length=self.output_length,
            output_signals=self.output_signals,
            embedding_map=self.embedding_map
        )

        
    def setup_data_preproc(self) -> None: 
        self.data_preproc = DataPreAndPostProcessor(
            input_signals=self.input_signals,
            data_sample_period=self.data_sample_period,
            interpolate_nan=self.interpolate_nan,
            zerofill_nan=self.zerofill_nan,
            input_preprocessing=self.input_preprocessing,
            datetime_embeddings=self.datetime_embeddings,
            output_signals=self.output_signals
        )


    def setup_model(self) -> None: 
        self.model = PredictionModel(
            input_length=self.input_length, 
            input_signals=self.input_signals,
            embedding_map=self.embedding_map,
            seq_layer_neurons=self.seq_layer_neurons,
            long_layer_neurons=self.long_layer_neurons,
            flat_layer_neurons=self.flat_layer_neurons,
            output_length=self.output_length,
            output_signals=self.output_signals,
            percentiles=self.percentiles,
            smoothing_weight=self.smoothing_weight
        )


    def model_summary(self) -> None: 
        self.model.print_summary()
        

    def fit(self, 
            data: list[pd.DataFrame],
            train_sample_n: int = None,
            test_sample_n: int = None,
            train_test_split: float = 0.8, 
            learning_rate: float = 1e-4, 
            batch_size: int = 64,
            epochs: int = 30,
            random_seed: int = None, 
            verbose: bool = True
           ) -> None:
        """
        Fit the entire package, i.e. preprocessing, neural network, 
        postprocessing and other possible parameters. 
        """
        self.train_sample_n = train_sample_n
        self.test_sample_n = test_sample_n
        self.train_test_split = train_test_split
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_seed = random_seed
        self.verbose = verbose
        
        # Reproducibility features
        if self.random_seed is not None: 
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)

        if self.fit_done == False: 
            self.logging('Setting up preprocessing')
            self.setup_data_preproc()
            self.data_preproc.fit(data)

        # Preprocess the data
        self.logging('Building dataset')
        proc_data = self.data_preproc.preprocess(data)
        
        if self.fit_done == False: 
            # Parse embedding enumeration from data
            self.embedding_map = self.data_preproc.parse_embedding_mapping(
                datetime_embeddings=self.datetime_embeddings, 
                datetime_embedding_dim=self.datetime_embedding_dim,
                categorical_embeddings=self.categorical_inputs, 
                categorical_embedding_dim=self.categorical_embedding_dim,
                data=proc_data,
            )        

        # Setup utils instance
        self.setup_model_utils()
        
        # Calculate number of rows in each dataframe
        df_lens = []
        for df in data: 
            df_lens.append(len(df))
        tot_len = sum(df_lens)
        
        # Create training dataset. Each dataframe is processed separately into train and test samples.
        train_inputs = []
        train_targets = []
        for i, df in enumerate(proc_data): 
            train_n = int(self.train_sample_n * df_lens[i] / tot_len)
            split_i = int(df_lens[i] * self.train_test_split)
            
            inputs, targets = self.model_utils.create_samples(
                data=df.iloc[0 : split_i], 
                sample_count=train_n,
                embedding_ood_ratio=self.embedding_ood_ratio,
            )
            train_inputs += inputs
            train_targets += targets
        
        # Convert training samples to dataloader format
        train_dataset = PredictionDataset(train_inputs, train_targets)
        train_loader = self.model_utils.create_data_loader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )

        # Create testing dataset. Each dataframe is processed separately into train and test samples.
        test_inputs = []
        test_targets = []
        for i, df in enumerate(proc_data): 
            test_n = int(self.test_sample_n * df_lens[i] / tot_len)
            split_i = int(df_lens[i] * self.train_test_split)
            
            inputs, targets = self.model_utils.create_samples(
                data=df.iloc[split_i :], 
                sample_count=test_n,
            )
            test_inputs += inputs
            test_targets += targets

        # Convert testing samples to dataloader format
        test_dataset = PredictionDataset(test_inputs, test_targets)
        test_loader = self.model_utils.create_data_loader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        # Setup the model
        if self.fit_done == False: 
            self.setup_model()
        
        # Fit the neural network
        self.model_trainer = PredictionModelTrainer(
            model=self.model, 
            learning_rate=self.learning_rate, 
            epochs=self.epochs, 
            logging=self.logging
        )
        self.logging('Training the model')
        self.train_losses, self.test_losses = self.model_trainer.fit(train_loader, test_loader)
        self.logging('Model training done')
        self.fit_done = True

        # Evaluate results
        self.evaluate(test_inputs, test_targets)
        
    
    def evaluate(self, test_inputs: list, test_targets: list) -> None: 

        self.results = {}
        for col in self.output_signals:
            self.results[col] = {}
            self.results[col]['abs_error'] = []

        for i in range(len(test_inputs)):
            prediction = self.model.predict(test_inputs[i])
            prediction = self.model_utils.parse_prediction_from_tensor(prediction)
            prediction = self.data_preproc.postprocess(prediction)
            prediction = prediction['prediction']
            
            ground_truth = self.model_utils.parse_prediction_from_tensor(test_targets[i])
            ground_truth = self.data_preproc.postprocess(ground_truth)
            ground_truth = ground_truth['ground_truth']

            for col in self.output_signals:
                pred_values = prediction[col].values
                gt_values = ground_truth[col].values
                abs_error = np.abs(pred_values - gt_values)
                self.results[col]['abs_error'].extend(list(abs_error))
                
        for col in self.output_signals:
            self.results[col]['MAE'] = np.mean(self.results[col]['abs_error']).astype(float)
            self.results[col].pop('abs_error')

        self.logging('Test results:')
        for col in self.output_signals: 
            self.logging('{}:'.format(col))
            self.logging('\tMAE: {:0.5f}'.format(self.results[col]['MAE']))
        
        
    def predict(self, data: pd.DataFrame, timestamp: pd.Timestamp = None) -> pd.DataFrame:
        """
        Inference mode prediction. 
        """
        # If prediction timestamp is not given, use the last timestamp in the data.
        if timestamp is None: 
            timestamp = data.index[-1]       
            
        # Get a slice of the data to avoid processing large amount of unnecessary rows
        i_end = data.index.get_indexer([timestamp], method='pad')[0]
        ts_end = data.index[i_end]
        est_ts_start = timestamp - timedelta(seconds=self.data_sample_period) * self.input_length
        i_start = data.index.get_indexer([est_ts_start], method='pad')[0] - 1
        ts_start = data.index[i_start]
        
        # Preprocess the data
        proc_data = self.data_preproc.preprocess(data.loc[ts_start : ts_end])[0]
        input_tensor = self.model_utils.create_input_tensor(proc_data, ts_end)
        
        # Make prediction and process it to original scale
        predictions = self.model.predict(input_tensor)
        predictions = self.model_utils.parse_prediction_from_tensor(predictions)
        predictions = self.data_preproc.postprocess(predictions)

        results = {}
        for key, values in predictions.items():
            # Add timestamps to prediction
            period = timedelta(seconds=self.data_sample_period)
            ts_index = pd.Series([ts_end + (1 + i) * period for i in range(self.output_length)])       
            values = values.set_index(ts_index)
            results[key] = values
        return results

    
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
        self.setup_model_utils()
        self.setup_data_preproc()
        self.setup_model()
        self.model.load(filename)
        self.data_preproc.set_params(params['preproc_params'])

        
    def logging(self, txt: str) -> None: 
        """
        Logging and printing. 
        """
        if self.verbose:
            print(txt)
        