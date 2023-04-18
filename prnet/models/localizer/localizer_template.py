import os
import torch
import torch.nn as nn
import numpy as np
from prnet.utils.params import ModelParams


class LocalizerTemplate(nn.Module):
    def __init__(self, model_params: ModelParams):
        super(LocalizerTemplate, self).__init__()
        self.model_params = model_params
        
    def forward(self, batch):
        img = batch['img']
        pc = batch['pc']

        # x is (batch_size, output_dim) tensor
        return {'global': x}


    def print_info(self):
        print('Model class: LocalizerTemplate')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))

