import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet101, resnet50
from prnet.utils.params import ModelParams

from prnet.models.attention.attentions import L2Attention
import numpy as np
import sys
sys.path.append("..")


class SmoothingAvgPooling(nn.Module):
    """Average pooling that smoothens the feature map, keeping its size

    :param int kernel_size: Kernel size of given pooling (e.g. 3)
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        pad = self.kernel_size // 2
        return F.avg_pool2d(x, (self.kernel_size, self.kernel_size), stride=1, padding=pad,
                        count_include_pad=False)


def pcawhitenlearn_shrinkage(X, s=1.0):
    """Learn PCA whitening with shrinkage from given descriptors"""
    N = X.shape[0]

    # Learning PCA w/o annotations
    m = X.mean(axis=0, keepdims=True)
    Xc = X - m
    Xcov = np.dot(Xc.T, Xc)
    Xcov = (Xcov + Xcov.T) / (2*N)
    eigval, eigvec = np.linalg.eig(Xcov)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    eigval = np.clip(eigval, a_min=1e-14, a_max=None)
    P = np.dot(np.linalg.inv(np.diag(np.power(eigval, 0.5*s))), eigvec.T)

    return m, P.T


class ConvDimReduction(nn.Conv2d):
    """Dimensionality reduction as a convolutional layer

    :param int input_dim: Network out_channels
    :param in dim: Whitening out_channels, for dimensionality reduction
    """

    def __init__(self, input_dim, dim):
        super().__init__(input_dim, dim, (1, 1), padding=0, bias=True)

    def initialize_pca_whitening(self, des):
        """Initialize PCA whitening from given descriptors. Return tuple of shift and projection."""
        m, P = whitening.pcawhitenlearn_shrinkage(des)
        m, P = m.T, P.T

        projection = torch.Tensor(P[:self.weight.shape[0], :]).unsqueeze(-1).unsqueeze(-1)
        self.weight.data = projection.to(self.weight.device)

        projected_shift = -torch.mm(torch.FloatTensor(P), torch.FloatTensor(m)).squeeze()
        self.bias.data = projected_shift[:self.weight.shape[0]].to(self.bias.device)
        return m.T, P.T


class HowNet(nn.Module):
    def __init__(self, model_params: ModelParams):
        super(HowNet, self).__init__()
        self.model_params = model_params
        
        if model_params.backbone_2d == 'res18':
            self.backbone_2d = resnet18(pretrained=True)
            self.input_dim = 512
        elif model_params.backbone_2d == 'res50':
            self.backbone_2d = resnet50(pretrained=True)
            self.input_dim = 2048
        else:
            raise NotImplementedError('Unknown backbone2d: {}'.format(model_params.backbone_2d))

        self.features = nn.Sequential(*list(self.backbone_2d.children())[:-2])
        self.attention = L2Attention()
        self.smoothing = SmoothingAvgPooling(kernel_size=3)
        self.dim_reduction = ConvDimReduction(self.input_dim, 128)


    def forward(self, batch):
        x = batch['img']

        feats = []
        masks = []
        o = self.features(x)
        m = self.attention(o)
        o = self.smoothing(o)
        o = self.dim_reduction(o)
        feats.append(o)
        masks.append(m)

        # Normalize max weight to 1
        mx = max(x.max() for x in masks)
        masks = [x/mx for x in masks]

        desc = torch.zeros((feats[0].shape[0], feats[0].shape[1]), dtype=torch.float32, device=feats[0].device)
        for feats, weights in zip(feats, masks):
            desc += (feats * weights.unsqueeze(-3)).sum((-2, -1)).squeeze()
        eps = 1e-6
        x = desc / (torch.norm(desc, p=2, dim=1, keepdim=True) + eps).expand_as(desc)

        # x is (batch_size, output_dim) tensor
        return {'global': x, 'feat': feats[0], 'mask': masks[0]}


    def print_info(self):
        print('Model class: HowNet')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))

