from prnet.models.localizer.fusion import FusionNet
from prnet.models.localizer.netvlad import NetVLAD_Pretrain
from prnet.models.localizer.netvlad import NetVLAD
from prnet.models.localizer.deformable import DeformableNet
from prnet.models.localizer.deformable_fusion import DeformAttnFusionNet
from prnet.models.localizer.hownet import HowNet
from prnet.models.localizer.dolgnet import DolgNet
from prnet.utils.params import ModelParams
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


def model_factory(model_params: ModelParams):

    if 'fusion_' in model_params.model:
        model = FusionNet(model_params)
    elif 'deformable' in model_params.model:
        model = DeformableNet(model_params)
    elif 'deformattn' in model_params.model:
        model = DeformAttnFusionNet(model_params)
    elif 'hownet' in model_params.model:
        model = HowNet(model_params)
    elif 'dolg' in model_params.model:
        model = DolgNet(model_params)
    elif 'netvlad' in model_params.model and not 'pretrain' in model_params.model:
        model = NetVLAD(model_params)
    elif 'netvlad' in model_params.model and 'pretrain' in model_params.model:
        model = build_netvlad_pretrain(model_params)
    else:
        raise NotImplementedError('Model not implemented: {}'.format(model_params.model))

    return model


class NetVLAD_finetune(nn.Module):
    def __init__(self, encoder, pool):
        super(NetVLAD_finetune, self).__init__()
        self.encoder = encoder
        self.pool = pool

    def forward(self, batch):
        x = batch["img"]
        x = self.encoder(x)
        x = self.pool(x)
        
        return {'global': x}


def build_netvlad_pretrain(model_params):

    print('===> Building NetVLAD model')
    encoder_dim = 512
    num_clusters = 64
    encoder = models.vgg16(pretrained=True)
    # capture only feature part and remove last relu and maxpool
    layers = list(encoder.features.children())[:-2]
    for l in layers[:-1]:
        for p in l.parameters():
            p.requires_grad = False
    layers.append(L2Norm())
    encoder = nn.Sequential(*layers)
    net_vlad = NetVLAD_Pretrain(num_clusters=64, dim=encoder_dim, vladv2=False)

    model = NetVLAD_finetune(encoder, net_vlad)

    return model
