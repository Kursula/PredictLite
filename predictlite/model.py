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
                 embeddings,
                 embedding_map,
                 hidden_layer_neurons,
                 hidden_layer_n, 
                 output_length,
                 output_signals
                ):
        super().__init__()
                
        self.embeddings = embeddings
        self.embedding_map = embedding_map
        self.input_length = input_length
        self.input_signals = input_signals
        self.output_length = output_length
        self.output_signals = output_signals
        
        # Define embedding inputs
        self.embedding_layers = []
        if embeddings is not None: 
            self.use_embeddings = True
            for col in embeddings:
                dim = self.embedding_map[col]['dim']
                # OOD embedding has always the highest value. 
                num = self.embedding_map[col]['ood'] + 1
                self.embedding_layers.append(nn.Embedding(num, dim))
        else: 
            self.use_embeddings = False
            
        # Define size of first hidden layer input
        input_size = input_length * len(input_signals)
        if self.use_embeddings: 
            for col in embeddings: 
                input_size += self.embedding_map[col]['dim'] 
        
        prev_layer_neuron_n = input_size

        # MLP layers
        self.mlp_layers = []
        for hid_size in hidden_layer_neurons:
            if hid_size is None: 
                this_layer_neuron_n = prev_layer_neuron_n
            else:
                this_layer_neuron_n = hid_size
            self.mlp_layers.append(nn.Linear(prev_layer_neuron_n, this_layer_neuron_n))
            self.mlp_layers.append(nn.ReLU())
            prev_layer_neuron_n = this_layer_neuron_n

        # Output
        output_size = output_length * len(output_signals)
        self.mlp_layers.append(nn.Linear(prev_layer_neuron_n, output_size))
        self.model = nn.Sequential(*self.mlp_layers)
            
            
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
            inputs = torch.cat(inputs, dim=-1)
            
        y_pred = self.model(inputs)    
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
        inputs = [x[0]]
        if self.use_embeddings:
            for i, emb_input in enumerate(x[1:]):
                emb_vec = self.embedding_layers[i](emb_input)
                emb_vec = torch.squeeze(emb_vec)
                inputs.append(emb_vec)
            inputs = torch.cat(inputs, dim=-1)

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(inputs.unsqueeze(0)).detach()[0]
        return y_pred


    def save(self, filename: str) -> None: 
        fn = filename.split('.')[0] + '.model'
        torch.save(self.state_dict(), fn)
            
    
    def load(self, filename: str) -> None: 
        fn = filename.split('.')[0] + '.model'
        self.load_state_dict(torch.load(fn))

        
    def print_summary(self) -> None: 
        print('Inputs:')
        for input_signal in self.input_signals: 
            print('\t{}: {}'.format(input_signal, self.input_length))
        if self.use_embeddings: 
            for i, emb in enumerate(self.embeddings):
                print('\t{}: {}'.format(emb, self.embedding_layers[i]))
                
        print('MLP:')
        for layer in self.mlp_layers: 
            print('\t{}'.format(layer))
            
        print('Outputs:')
        for output_signal in self.output_signals: 
            print('\t{}: {}'.format(output_signal, self.output_length))

