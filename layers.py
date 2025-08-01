import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict



class MLP(nn.Module):
    def __init__(self, hidden_dim=[1000, 2048, 512], act=nn.Tanh(), dropout=0.01):
        super(MLP, self).__init__()
        #flickr 0,01
        #nus 0.4
        #coco 0.5

        self.input_dim = hidden_dim[0]
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim[-1]

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.batch_norms = nn.ModuleList()  

        # Add input layer
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim[0]))
        self.activations.append(act)
        self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim[0]))

        # Add hidden layers
        for i in range(len(self.hidden_dim) - 1):
            self.layers.append(nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]))
            self.activations.append(act)
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim[i + 1])) 

        # Add output layer
        self.layers.append(nn.Linear(self.hidden_dim[-1], self.output_dim))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for layer, activation, batch_norm in zip(self.layers, self.activations, self.batch_norms):
            x = layer(x)
            x = activation(x)
            x = batch_norm(x)
            x = self.dropout(x)

        return x


