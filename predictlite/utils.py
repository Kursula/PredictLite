################################################################################
# This is PredictLite, a lightweight time series prediction model using        
# PyTorch and basic Numpy and Pandas processing functions.
#
# Copyright Mikko Kursula 2022 - 2024. MIT license. 
################################################################################

# General
import pandas as pd
import numpy as np

# Pytorch
import torch
import torch.nn as nn

        
class PredictionDataset(torch.utils.data.Dataset):
    
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
            
            
    def __len__(self):
        return len(self.targets)
    
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class PredictionModelUtils:

    def __init__(self, 
                 input_length: int, 
                 input_signals: list,
                 output_length: int, 
                 output_signals: list,
                 embedding_map: dict,
                ):
        self.input_length = input_length
        self.input_signals = input_signals
        self.output_length = output_length 
        self.output_signals = output_signals
        self.embedding_map = embedding_map

    
    def create_input_tensor(self, 
                            data: pd.DataFrame, 
                            timestamp: pd.Timestamp, 
                            force_ood: bool = False
                           ) -> torch.tensor: 
        """
        Covert dataframe data to a Torch tensor in model input configuration. 
        Timestamp gives the last row in input data. 
        """
        input_tensors = []

        # Start and end index values
        i_end = data.index.get_indexer([timestamp], method='pad')[0]
        i_start = i_end - self.input_length + 1
        ts_start = data.index[i_start]
        ts_end = timestamp
        
        # Float values
        input_values = data.loc[ts_start : ts_end, self.input_signals].values #.flatten(order='F')
        input_tensors.append(torch.from_numpy(input_values).float())
        
        # Embeddings 
        if self.embedding_map is not None:
            for col in self.embedding_map.keys():
                emb_keys = []
                for value in data.loc[ts_start : ts_end, col].astype(str).values.tolist(): 
                    if (value in self.embedding_map[col]['map'].keys()) and (force_ood == False):
                        emb_key = self.embedding_map[col]['map'][value]
                    else: 
                        emb_key = self.embedding_map[col]['ood']
                    emb_keys.append(emb_key)
                input_tensors.append(torch.LongTensor(emb_keys))
            
        return input_tensors
    
    
    def create_target_tensor(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> torch.tensor: 
        """
        Covert dataframe data to a Torch tensor in model output configuration. 
        Timestamp gives the last row in input data. 
        """
        i_start = data.index.get_indexer([timestamp], method='pad')[0] + 1
        ts_start = data.index[i_start] 
        i_end = i_start + self.output_length - 1
        ts_end = data.index[i_end]
        target_values = data.loc[ts_start : ts_end, self.output_signals].values.flatten(order='F')
        target_tensor = torch.from_numpy(target_values).float()                          
        return target_tensor
    
    
    def parse_prediction_from_tensor(self, output_tensor: torch.tensor) -> pd.DataFrame:
        results = {}
        if len(output_tensor.shape) == 3: 
            keys = ['prediction', 'lower_percentile', 'upper_percentile']
            for i in range(len(keys)):
                values = output_tensor[:, i, :].numpy().reshape((self.output_length, len(self.output_signals)), order='F')
                pred = pd.DataFrame(data=values, columns=self.output_signals)
                results[keys[i]] = pred
        else:
            # Used only to convert ground truth back to original data format. 
            values = output_tensor.numpy().reshape((self.output_length, len(self.output_signals)), order='F')
            pred = pd.DataFrame(data=values, columns=self.output_signals)
            results['ground_truth'] = pred
        return results

    
    def create_samples(self, data: pd.DataFrame, sample_count: int, embedding_ood_ratio: float = 0) -> list:
        """
        Create input and target sample pairs for model training.
        embedding_ood_ratio: ratio of samples that will be forced to use ood (out of distribution)
        embeddings in order to train those embeddings. 
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
            
        # Create samples
        for timestamp in sample_timestamps:
            if np.random.rand() < embedding_ood_ratio: 
                force_ood = True
            else: 
                force_ood = False
                
            input_tensor = self.create_input_tensor(data, timestamp, force_ood)
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
    
    


