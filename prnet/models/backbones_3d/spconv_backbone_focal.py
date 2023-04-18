from functools import partial

import torch
from prnet.utils.spconv_utils import spconv
import torch.nn as nn

from .focal_sparse_conv.focal_sparse_conv import FocalSparseConv
from .focal_sparse_conv.SemanticSeg.pyramid_ffn import PyramidFeat2D


class objDict:
    @staticmethod
    def to_object(obj: object, **data):
        obj.__dict__.update(data)

class ConfigDict:
    def __init__(self, name):
        self.name = name
    def __getitem__(self, item):
        return getattr(self, item)


class SparseSequentialBatchdict(spconv.SparseSequential):
    def __init__(self, *args, **kwargs):
        super(SparseSequentialBatchdict, self).__init__(*args, **kwargs)

    def forward(self, input, batch_dict=None):
        loss = 0
        for k, module in self._modules.items():
            if module is None:
                continue
            if isinstance(module, (FocalSparseConv,)):
                input, batch_dict = module(input, batch_dict)
            else:
                input = module(input)
        return input, batch_dict, loss


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(True),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out


class VoxelBackBone8xFocal(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(True),
        )

        block = post_act_block

        topk = True
        threshold = 0.5
        kernel_size = 3
        mask_multi = False
        skip_mask_kernel = False
        skip_mask_kernel_image = False
        enlarge_voxel_channels = -1

        special_spconv_fn = partial(FocalSparseConv, mask_multi=mask_multi, enlarge_voxel_channels=enlarge_voxel_channels, 
                                    topk=topk, threshold=threshold, kernel_size=kernel_size, padding=kernel_size//2, 
                                    skip_mask_kernel=skip_mask_kernel)

        self.conv1 = SparseSequentialBatchdict(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
            special_spconv_fn(16, 16, voxel_stride=1, norm_fn=norm_fn, indice_key='focal1'),
        )

        self.conv2 = SparseSequentialBatchdict(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=(1, 1, 1), padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            special_spconv_fn(32, 32, voxel_stride=2, norm_fn=norm_fn, indice_key='focal2'),
        )

        self.conv3 = SparseSequentialBatchdict(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=(1, 1, 1), padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            special_spconv_fn(64, 64, voxel_stride=4, norm_fn=norm_fn, indice_key='focal3'),
        )

        self.conv4 = SparseSequentialBatchdict(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=(1, 1, 1), padding=(1, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        if model_cfg.use_rgb and model_cfg.use_xyz:
            self.conv_out = spconv.SparseSequential(
                # [200, 150, 5] -> [200, 150, 2]
                spconv.SparseConv3d(64, model_cfg.feature_dim//2, (2, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                                    bias=False, indice_key='spconv_down2'),
                norm_fn(model_cfg.feature_dim//2),
                nn.ReLU(),
            )
            self.num_point_features = model_cfg.feature_dim//2
        else:
            self.conv_out = spconv.SparseSequential(
                # [200, 150, 5] -> [200, 150, 2]
                spconv.SparseConv3d(64, model_cfg.feature_dim, (2, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                                    bias=False, indice_key='spconv_down2'),
                norm_fn(model_cfg.feature_dim),
                nn.ReLU(),
            )
            self.num_point_features = model_cfg.feature_dim

        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }
        

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1, batch_dict, loss1 = self.conv1(x, batch_dict)

        x_conv2, batch_dict, loss2 = self.conv2(x_conv1, batch_dict)
        x_conv3, batch_dict, loss3 = self.conv3(x_conv2, batch_dict)
        x_conv4, batch_dict, loss4 = self.conv4(x_conv3, batch_dict)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict
