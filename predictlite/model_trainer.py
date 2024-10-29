################################################################################
# This is PredictLite, a lightweight time series prediction model using        
# PyTorch and basic Numpy and Pandas processing functions.
#
# Copyright Mikko Kursula 2022 - 2024. MIT license. 
################################################################################

# General
import numpy as np
from typing import Callable

# Pytorch
import torch
import torch.nn as nn

# PredictLite modules 
from predictlite.model import PredictionModel


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
        test_mape = []
        
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

