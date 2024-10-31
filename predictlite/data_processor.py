################################################################################
# This is PredictLite, a lightweight time series prediction model using        
# PyTorch and basic Numpy and Pandas processing functions.
#
# Copyright Mikko Kursula 2022 - 2024. MIT license. 
################################################################################

# General
import pandas as pd


def combine_dfs(data: list[pd.DataFrame], column: str) -> pd.DataFrame: 
    if len(data) == 1: 
        return data[0][column]
    
    combined_data = pd.concat([df[column] for df in data], ignore_index=True)
    return combined_data


class DataPreAndPostProcessor:

    proc_options = ['minmax', 'z-norm', 'none']
    datetime_emb_options = ['year', 'month', 'day', 'day_of_week', 'hour', 'minute', 'second']
    
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
        
        
    def fit(self, data: list[pd.DataFrame]) -> None: 
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
            combined_data = combine_dfs(data, col)
            
            self.preproc_stats['signal_min_values'][col] = combined_data.dropna().min()
            self.preproc_stats['signal_max_values'][col] = combined_data.dropna().max()
            self.preproc_stats['signal_mean_values'][col] = combined_data.dropna().mean()
            self.preproc_stats['signal_std_values'][col] = combined_data.dropna().std()

    
    def preprocess(self, data: list[pd.DataFrame]) -> list[pd.DataFrame]: 
        """
        Preprocess the data.
        """
        if type(data) != list: 
            data = [data]
        proc_data = []
        for df in data: 
            proc_df = df.copy()
            proc_df = proc_df.sort_index()
        
            # Process NaN values
            if self.interpolate_nan: 
                proc_df[self.input_signals] = proc_df[self.input_signals].interpolate()
            if self.zerofill_nan:
                proc_df[self.input_signals] = proc_df[self.input_signals].fillna(0)
        
            # Scale values
            for col in self.input_signals: 
                if self.input_preprocessing[col] == 'minmax':
                    min_val = self.preproc_stats['signal_min_values'][col]
                    max_val = self.preproc_stats['signal_max_values'][col]
                    proc_df[col] = (proc_df[col] - min_val) / (max_val - min_val)
                
                elif self.input_preprocessing[col] == 'z-norm':
                    mean_val = self.preproc_stats['signal_mean_values'][col]
                    std_val = self.preproc_stats['signal_std_values'][col]
                    proc_df[col] = (proc_df[col] - mean_val) / (std_val + self.eps)
                    
            # Datetime embeddings
            for emb in self.datetime_embeddings: 
                if emb == 'year':
                    proc_df['year_emb'] = proc_df.index.year
                elif emb == 'month':
                    proc_df['month_emb'] = proc_df.index.month
                elif emb == 'day':
                    proc_df['day_emb'] = proc_df.index.day
                elif emb == 'day_of_week':
                    proc_df['day_of_week_emb'] = proc_df.index.dayofweek
                elif emb == 'hour':
                    proc_df['hour_emb'] = proc_df.index.hour
                elif emb == 'minute':
                    proc_df['minute_emb'] = proc_df.index.minute
                elif emb == 'second':
                    proc_df['second_emb'] = proc_df.index.second
            
            proc_data.append(proc_df)
            
        return proc_data
    
    
    def postprocess(self, data: dict) -> dict: 
        """
        Post-process the prediction results, i.e. scale the values back to original scale. 
        Outputs share the same preprocessing with inputs. 
        Post-processing is done for one prediction at a time. 
        """
        results = {}
        for key, values in data.items(): 
            proc_data = values.copy()
            
            for col in self.output_signals:
                if self.input_preprocessing[col] == 'minmax':
                    min_val = self.preproc_stats['signal_min_values'][col]
                    max_val = self.preproc_stats['signal_max_values'][col]
                    proc_data[col] = proc_data[col] * (max_val - min_val) + min_val
                
                elif self.input_preprocessing[col] == 'z-norm':
                    mean_val = self.preproc_stats['signal_mean_values'][col]
                    std_val = self.preproc_stats['signal_std_values'][col]
                    proc_data[col] = proc_data[col] * std_val + mean_val
            
            results[key] = proc_data
    
        return results

    def parse_embedding_mapping(self,
                                datetime_embeddings: list, 
                                datetime_embedding_dim: int, 
                                categorical_embeddings: list, 
                                categorical_embedding_dim: int,
                                data: list[pd.DataFrame],
                               ) -> dict:
        """
        Create map of categorical values to embedding index values. 
        Out of distribution (ood) index is set for values that do not exist in training data. 
        """
        if (len(datetime_embeddings) == 0) and (len(categorical_embeddings) == 0): 
            return None
        
        emb_map = {}
        for col in datetime_embeddings: 
            emb_col_name = '{}_emb'.format(col)
            emb_map[emb_col_name] = {'map' : {}, 'ood' : None, 'dim' : datetime_embedding_dim}

            combined_data = combine_dfs(data, emb_col_name)
            
            for i, value in enumerate(combined_data.unique()):
                emb_map[emb_col_name]['map'][str(value)] = i 
            
            emb_map[emb_col_name]['ood'] = i + 1
            
        for col in categorical_embeddings: 
            emb_map[col] = {'map' : {}, 'ood' : None, 'dim' : categorical_embedding_dim}
            
            combined_data = combine_dfs(data, col)
            
            for i, value in enumerate(combined_data.unique()):
                emb_map[col]['map'][str(value)] = i 
            emb_map[col]['ood'] = i + 1
            
        return emb_map

    
    def get_params(self) -> dict: 
        return self.preproc_stats
    
    
    def set_params(self, params: dict) -> None: 
        self.preproc_stats = params
