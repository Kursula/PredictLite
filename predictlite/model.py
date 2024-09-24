################################################################################
# This is PredictLite, a lightweight time series prediction model using        
# PyTorch and basic Numpy and Pandas processing functions.
#
# Copyright Mikko Kursula 2022. MIT license. 
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
                 output_signals
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

        self.setup_nn()

    
    def setup_nn(self) -> None:
        activation = nn.Mish()
        input_size = 0
        
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
                input_size += dim
        else: 
            self.use_embeddings = False

        # Calculate total input width
        input_size += len(self.input_signals)
        
        # Sequential MLP layers 
        self.seq_layers = nn.ModuleList()
        prev_layer_neuron_n = input_size
        for neuron_n in self.seq_layer_neurons:
            self.seq_layers.append(nn.Linear(prev_layer_neuron_n, neuron_n))
            self.seq_layers.append(activation)
            prev_layer_neuron_n = neuron_n

        self.seq_layers.append(nn.Flatten(start_dim=-2))
        seq_output_n = prev_layer_neuron_n * self.input_length

        # Longitudinal MLP layers 
        self.long_layers = nn.ModuleList()
        prev_layer_neuron_n = self.input_length
        for neuron_n in self.long_layer_neurons:
            self.long_layers.append(nn.Linear(prev_layer_neuron_n, neuron_n))
            self.long_layers.append(activation)
            prev_layer_neuron_n = neuron_n

        self.long_layers.append(nn.Flatten(start_dim=-2))
        long_output_n = prev_layer_neuron_n * input_size
        
        # Flattened MLP layers
        prev_layer_neuron_n = seq_output_n + long_output_n
        self.flat_layers = nn.ModuleList()
        for neuron_n in self.flat_layer_neurons:
            self.flat_layers.append(nn.Linear(prev_layer_neuron_n, neuron_n))
            self.flat_layers.append(activation)
            prev_layer_neuron_n = neuron_n

        # Output
        output_size = self.output_length * len(self.output_signals)
        self.flat_layers.append(nn.Linear(prev_layer_neuron_n, output_size))
                    
            
    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Training forward pass. 
        Not used in inference phase. 
        """        
        inputs = [x[0]]
        if self.use_embeddings:
            for i, emb_input in enumerate(x[1:]):
                emb_vec = self.embedding_layers[i](emb_input)
                emb_vec = torch.squeeze(emb_vec)
                inputs.append(emb_vec)
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
        
        # Flattened layers
        x = torch.cat([seq_out, long_out], dim=-1)
        for layer in self.flat_layers: 
            x = layer(x)

        return x
        #y_pred = self.model(inputs)    
        #return y_pred
    
    
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
        
        print('MLP flattened part:')
        for layer in self.flat_layers: 
            if type(layer) == torch.nn.modules.linear.Linear: 
                pc = (layer.in_features + 1) * layer.out_features
                tot_mlp_params += pc
                print('\t{}, parameter count: {}'.format(layer, pc))
            else: 
                print('\t{}'.format(layer))        
                
        print('\nTotal MLP parameter count: {}'.format(tot_mlp_params))
        
