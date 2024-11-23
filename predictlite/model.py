################################################################################
# This is PredictLite, a lightweight time series prediction model using        
# PyTorch and basic Numpy and Pandas processing functions.
#
# Copyright Mikko Kursula 2022 - 2024. MIT license. 
################################################################################

# General
import numpy as np
import pandas as pd

# Pytorch
import torch
import torch.nn as nn

        
class PredictionModel(nn.Module):

    def __init__(self, 
                 input_length, 
                 input_signals,
                 embedding_map,
                 seq_layer_neurons,
                 long_layer_neurons,
                 flat_layer_neurons,
                 output_length,
                 output_signals,
                 percentiles, 
                 smoothing_weight, 
                ):
        super().__init__()
                
        self.input_length = input_length
        self.input_signals = input_signals
        self.embedding_map = embedding_map
        self.long_layer_neurons = long_layer_neurons
        self.seq_layer_neurons = seq_layer_neurons
        self.flat_layer_neurons = flat_layer_neurons
        self.output_length = output_length
        self.output_signals = output_signals
        self.output_dim = output_length * len(output_signals)
        self.output_signal_n = len(output_signals)
        self.lower_p = min(percentiles)
        self.upper_p = max(percentiles)
        self.smoothing_weight = smoothing_weight
        
        self.setup_nn()

    
    def setup_nn(self) -> None:
        activation = nn.ReLU()
        input_dim = 0
        
        # Define embedding inputs
        self.embedding_layers = nn.ModuleList()
        if self.embedding_map is not None: 
            self.use_embeddings = True
            for col in self.embedding_map.keys():
                dim = self.embedding_map[col]['dim']
                num = self.embedding_map[col]['ood'] + 1 # OOD embedding has always the highest value. 
                emb_layer = nn.Embedding(num, dim)
                emb_layer.predlite_name = col # give a name to the embedding module
                self.embedding_layers.append(emb_layer)
                input_dim += dim
        else: 
            self.use_embeddings = False

        # Calculate total input width
        input_dim += len(self.input_signals)
        
        # Sequential MLP layers 
        self.seq_layers = nn.ModuleList()
        prev_layer_dim = input_dim
        for hidden_dim in self.seq_layer_neurons:
            self.seq_layers.append(nn.Linear(prev_layer_dim, hidden_dim))
            self.seq_layers.append(activation)
            prev_layer_dim = hidden_dim

        self.seq_layers.append(nn.Flatten(start_dim=-2))
        seq_output_dim = prev_layer_dim * self.input_length

        # Longitudinal MLP layers 
        self.long_layers = nn.ModuleList()
        prev_layer_dim = self.input_length
        for hidden_dim in self.long_layer_neurons:
            self.long_layers.append(nn.Linear(prev_layer_dim, hidden_dim))
            self.long_layers.append(activation)
            prev_layer_dim = hidden_dim

        self.long_layers.append(nn.Flatten(start_dim=-2))
        long_output_dim = prev_layer_dim * input_dim

        # First set of flattened MLP layers for coarse prediction
        prev_layer_dim = seq_output_dim + long_output_dim
        self.flat_layers1 = nn.ModuleList()
        self.flat_layers1.append(nn.Linear(prev_layer_dim, self.output_dim))
        self.flat_layers1.append(activation)

        # Second set of flattened MLP layers for refining the prediction
        prev_layer_dim = self.output_dim + seq_output_dim + long_output_dim
        self.flat_layers2 = nn.ModuleList()
        for hidden_dim in self.flat_layer_neurons:
            self.flat_layers2.append(nn.Linear(prev_layer_dim, hidden_dim))
            self.flat_layers2.append(activation)
            prev_layer_dim = hidden_dim

        # Output: 3x multiplier because the output contains: 
        # 0: predicted values 
        # 1: lower percentiles
        # 2: upper percentiles
        self.flat_layers2.append(nn.Linear(prev_layer_dim, 3 * self.output_dim))
                    
            
    def forward(self, x: torch.tensor) -> torch.tensor:
        
        inputs = [x[0]]

        if self.use_embeddings:
            for i, emb_keys in enumerate(x[1:]):
                emb_tensor = self.embedding_layers[i](emb_keys)
                inputs.append(emb_tensor)

        x_in = torch.cat(inputs, dim=-1)

        # Sequential layers
        x = x_in
        for layer in self.seq_layers: 
            x = layer(x)
        seq_out = x        

        # Longitudinal layers
        x = torch.rot90(x_in, 1, [-2, -1])
        for layer in self.long_layers: 
            x = layer(x)
        long_out = x
        
        # Combine tensors for flat layer processing
        x = torch.cat([seq_out, long_out], dim=-1)

        # Coarse prediction
        for layer in self.flat_layers1: 
            x = layer(x)
        coarse_output = x.reshape(-1, 1, self.output_dim)

        x = torch.cat([seq_out, long_out, x], dim=-1)

        # Refined prediction
        for layer in self.flat_layers2: 
            x = layer(x)

        # Reshape to [batch, 3, output_dim]
        refined_output = x.reshape(-1, 3, self.output_dim)  

        # Combine all outputs
        output = torch.cat((refined_output, coarse_output), dim=1)

        return output
    
    
    def loss(self, y_pred, y_true, progress=1.0): 
        # A hybrid percentile and smoothing loss

        # Curriculum learning mask
        horizon = self.output_length if progress >= 1 else max(1, round(self.output_length * progress))
        seq_mask = torch.zeros(self.output_length)
        seq_mask[:horizon] = 1 
        mask = seq_mask.view(1, self.output_length).repeat(1, self.output_signal_n)
        
        total_loss = 0

        # Parse predictions
        y_refined_pred = y_pred[:, 0, :]   # Prediction
        y_lower_pred = y_pred[:, 1, :]  # lower percentile
        y_upper_pred = y_pred[:, 2, :]  # upper percentile 
        y_coarse_pred = y_pred[:, 3, :]  # coarse prediction

        # Refined prediction loss.
        # The median of value distribution is used as prediction. 
        percentile = 0.5
        errors = y_refined_pred - y_true
        loss = torch.max(percentile * errors, (percentile - 1) * errors)
        loss = loss * mask
        total_loss += loss.mean()

        # Coarse prediction loss 
        percentile = 0.5
        errors = y_coarse_pred - y_true
        loss = torch.max(percentile * errors, (percentile - 1) * errors)
        loss = loss * mask
        total_loss += loss.mean()

        # Lower percentile line loss 
        errors = y_true - y_lower_pred
        loss = torch.max(self.lower_p * errors, (self.lower_p - 1) * errors)
        loss = loss * mask
        total_loss += loss.mean()
        
        # Upper percentile line loss
        errors = y_true - y_upper_pred
        loss = torch.max(self.upper_p * errors, (self.upper_p - 1) * errors)
        loss = loss * mask
        total_loss += loss.mean()

        # Smoothness losses
        tikhonov_loss_y_refi = self.smoothing_weight * self.tikhonov_regularization(y_refined_pred, seq_mask)
        tikhonov_loss_lower = self.smoothing_weight * self.tikhonov_regularization(y_lower_pred, seq_mask)
        tikhonov_loss_upper = self.smoothing_weight * self.tikhonov_regularization(y_upper_pred, seq_mask)
        total_loss += tikhonov_loss_y_refi + tikhonov_loss_lower + tikhonov_loss_upper

        return total_loss

    
    def tikhonov_regularization(self, y_pred, mask):
        
        # Reshape y_pred to [batch, output_vars, time_steps]
        y_pred_reshaped = y_pred.view(y_pred.size(0), len(self.output_signals), self.output_length)

        # Reshape mask 
        mask = mask.view(1, 1, 1, self.output_length)
        
        # Calculate differences along the time steps dimension
        diff = y_pred_reshaped[..., 1:] - y_pred_reshaped[..., :-1]
        diff = diff * mask[..., :-1]
        
        # Compute the squared differences and mean over all dimensions
        tikhonov_loss = (diff ** 2).mean()
        return tikhonov_loss

    
    def predict(self, x: torch.tensor) -> torch.tensor: 

        if len(x[0].shape) == 2: 
            for i in range(len(x)):
                x[i] = x[i].unsqueeze(0)

        self.eval()
        with torch.no_grad():
            y_pred = self.forward(x).detach()
        return y_pred


    def save(self, filename: str) -> None: 
        fn = filename.split('.')[0] + '.model'
        torch.save(self.state_dict(), fn)
            
    
    def load(self, filename: str) -> None: 
        fn = filename.split('.')[0] + '.model'
        self.load_state_dict(torch.load(fn))

        
    def print_summary(self) -> None: 
        tot_mlp_params = 0 
        
        print('Inputs:')
        print('\tFloat inputs: {}'.format(self.input_signals))
        print('\tTime steps: {}'.format(self.input_length))
        
        for layer in self.embedding_layers:
            print('\t{}: num_embeddings: {}, embedding_dim: {}'.format(layer.predlite_name, layer.num_embeddings, layer.embedding_dim))
        
        print('MLP sequential part:')
        for layer in self.seq_layers: 
            if type(layer) == torch.nn.modules.linear.Linear: 
                pc = (layer.in_features + 1) * layer.out_features
                tot_mlp_params += pc
                print('\t{}, parameter count: {}'.format(layer, pc))
            else: 
                print('\t{}'.format(layer))
        
        print('MLP longitudinal part:')
        for layer in self.long_layers: 
            if type(layer) == torch.nn.modules.linear.Linear: 
                pc = (layer.in_features + 1) * layer.out_features
                tot_mlp_params += pc
                print('\t{}, parameter count: {}'.format(layer, pc))
            else: 
                print('\t{}'.format(layer))
        
        print('MLP flattened part for coarse prediction:')
        for layer in self.flat_layers1: 
            if type(layer) == torch.nn.modules.linear.Linear: 
                pc = (layer.in_features + 1) * layer.out_features
                tot_mlp_params += pc
                print('\t{}, parameter count: {}'.format(layer, pc))
            else: 
                print('\t{}'.format(layer))
                
        print('MLP flattened part for refined prediction:')
        for layer in self.flat_layers2: 
            if type(layer) == torch.nn.modules.linear.Linear: 
                pc = (layer.in_features + 1) * layer.out_features
                tot_mlp_params += pc
                print('\t{}, parameter count: {}'.format(layer, pc))
            else: 
                print('\t{}'.format(layer))                     
        print('\nTotal MLP parameter count: {}'.format(tot_mlp_params))
        
