# %%
import math
import typing as ty
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import copy
from tqdm import tqdm

import torchmetrics
device = torch.device('cuda')

""""
This model is heavilty adapted from an optuna example of resnet found at:
https://github.com/Yura52/tabular-dl-revisiting-models/blob/main/bin/resnet.py

Consider the original repo there for more information.

The main changes consist of the removal of embedding things for categorical features and some other 
ease-of-use stuff.
"""
class ResNet(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        d: int,
        d_hidden_factor: float,
        n_layers: int,
        activation = 'ReLU',
        normalization = 'batchnorm',
        hidden_dropout: float,
        residual_dropout: float,
        d_out: int,
    ) -> None:
        super().__init__()

        def make_normalization():
            return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
                normalization
            ](d)

        if activation == 'ReLU':
            self.main_activation = nn.ReLU()
            self.last_activation = nn.ReLU()

        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout

        d_in = d_numerical
        d_hidden = int(d * d_hidden_factor)

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'norm': make_normalization(),
                        'linear0': nn.Linear(
                            d, d_hidden * (2 if activation.endswith('glu') else 1)
                        ),
                        'linear1': nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, d_out)

    def forward(self, x_num: Tensor) -> Tensor:
        x = x_num
        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x

"""
The Training-Validation loop of the model.
Since this is tabular deep-learning and we are typically working with small datasets, the loop assumes that all
data has already been transferred to the GPU device. This dramatically speeds up training as it does not need to
copy data to the GPU and back every batch. In principle one could use one batch for all data, but this gives some
issues with the batch size/learning-rate interaction.
"""

def train_model(model, dataloaders, criterion, optimizer, dataset_sizes, num_epochs=1000, phases= ['train','val']):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    early_stopping_patience = 50
    min_epochs = 30
    best_epoch = 0
    auroc = torchmetrics.AUROC(task="binary")
    result_dict = {}
    for phase in ['train', 'val','test']:
        result_dict[phase+'_loss'] = []
        result_dict[phase+'_auc'] = []
    
    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            y_pred = []
            y_true = []
            
            # for inputs, labels in dataloaders[phase]:
            for inputs, labels in dataloaders[phase]:
                
                # zero the parameter gradients
                for param in model.parameters():
                    param.grad = None

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.detach().item() * inputs.detach().size(0)

                # Append labels and predictions to the epoch lists
                y_pred.append(outputs.detach())
                y_true.append(labels.detach())
            
            ###  GET METRICS FOR THIS EPOCH ####
            epoch_loss = running_loss / dataset_sizes[phase]
            
            # transfer epoch lists to tensors
            y_pred = torch.cat(y_pred).detach()
            y_true = torch.cat(y_true).detach()

            # Calculate epoch auc
            epoch_auc = auroc(y_pred, y_true).detach().item()

            # Append the results of the current epoch to the result dict
            # This can later be used to plot training and valdiation loss & auc
            result_dict[phase+'_loss'].append(epoch_loss)
            result_dict[phase+'_auc'].append(epoch_auc)

            # Keep track of the best epoch
            if phase == 'val' and epoch_auc > best_auc:
                best_epoch = epoch
                best_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        # At the end of each epoch, monitor early stopping
        if epoch > min_epochs and epoch-best_epoch > early_stopping_patience:
            # print('early stopping...')
            break
    
    #add the best val metric to the results dict
    result_dict['best_val_auc'] = best_auc
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model,result_dict

def evaluate(model, X_test, y_test):
    y_pred = model(X_test)
    auroc = torchmetrics.AUROC(task="binary")
    auc = auroc(y_pred, y_test).item()
    return auc