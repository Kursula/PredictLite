################################################################################
# This is PredictLite, a lightweight time series prediction model using        
# PyTorch and basic Numpy and Pandas processing functions.
#
# Copyright Mikko Kursula 2022. MIT license. 
################################################################################

# General
import pandas as pd


class DataPreAndPostProcessor:

    proc_options = ['minmax', 'z-norm', 'none']
    datetime_emb_options = ['year', 'month', 'day_of_week', 'hour', 'minute']
    
    def __init__(self,
                 input_signals: list,
                 data_sample_period: float,
                 interpolate_nan: bool,
                 zerofill_nan: bool,
                 input_preprocessing: dict, 
                 datetime_embeddings: list, 
                 output_signals: list,
                ):

        self.input_signals = input_signals
        self.interpolate_nan = interpolate_nan
        self.zerofill_nan = zerofill_nan
        self.input_preprocessing = input_preprocessing
        self.datetime_embeddings = datetime_embeddings
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
        proc_data = proc_data.sort_index()
        
        # Process NaN values
        if self.interpolate_nan: 
            proc_data[self.input_signals] = proc_data[self.input_signals].interpolate()
        if self.zerofill_nan:
            proc_data[self.input_signals] = proc_data[self.input_signals].fillna(0)
        
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
                
        # Datetime embeddings
        for emb in self.datetime_embeddings: 
            if emb == 'year':
                proc_data[emb] = proc_data.index.year
            if emb == 'month':
                proc_data[emb] = proc_data.index.month
            if emb == 'day_of_week':
                proc_data[emb] = proc_data.index.dayofweek
            if emb == 'hour':
                proc_data[emb] = proc_data.index.hour
            if emb == 'minute':
                proc_data[emb] = proc_data.index.minute
            
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
    
