from __future__ import print_function
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import transforms, utils
import prnet.utils.vox_utils.misc as misc
import prnet.utils.vox_utils.improc as improc
import prnet.utils.vox_utils.vox as vox
import prnet.utils.vox_utils.geom as geom
import prnet.utils.vox_utils.basic as basic
from prnet.utils.common_utils import ssc_to_homo
from mmcv.ops import Voxelization
from mmcv.runner import force_fp32
from prnet.models.backbones_2d.resnet import ResNet, BasicBlock
from prnet.models.backbones_3d import vfe
from prnet.models.aggregation.NetVLADLoupe import *
from prnet.utils.spconv_utils import find_all_spconv_keys
from .. import backbones_2d, backbones_3d
from ..backbones_3d import pfe, vfe
import torchvision
from torchvision.models.resnet import resnet18, resnet101, resnet50
import matplotlib.pyplot as plt
from prnet.utils.params import ModelParams
from efficientnet_pytorch import EfficientNet
from spconv.pytorch.utils import PointToVoxel, gather_features_by_pc_voxel_id
import pickle
import cv2

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def fftshift2d(x):
    for dim in range(1, len(x.size())):
        n_shift = x.size(dim)//2
        if x.size(dim) % 2 != 0:
            n_shift = n_shift + 1  # for odd-sized images
        x = roll_n(x, axis=dim, n=n_shift)
    return x  # last dim=2 (real&imag)


class FusionNet(torch.nn.Module):
    def __init__(self, model_params: ModelParams):
        super(FusionNet, self).__init__()
        self.feature_dim = model_params.feature_dim
        self.output_dim = model_params.output_dim
        self.Z, self.Y, self.X = model_params.Z, model_params.Y, model_params.X
        self.use_rgb = model_params.use_rgb
        self.use_xyz = model_params.use_xyz
        self.use_normalize = model_params.normalize
        self.H = model_params.H
        self.W = model_params.W
        self.aggregation = model_params.aggregation
        self.theta = model_params.theta
        self.radius = model_params.radius
        self.model_params = model_params
        self.coordinates = model_params.coordinates
        self.image_feat_dim = self.feature_dim
        self.use_cam_id = model_params.cam_id

        XMIN = model_params.xbounds[0]
        XMAX = model_params.xbounds[1]
        YMIN = model_params.ybounds[0]
        YMAX = model_params.ybounds[1]        
        ZMIN = model_params.zbounds[0]
        ZMAX = model_params.zbounds[1]

        if self.use_xyz:
            self.quantization_step = model_params.quantization_step

            if self.coordinates == 'polar':
                self.point_cloud_range = [0, YMIN/self.quantization_step[1], 0, XMAX/self.quantization_step[0], YMAX/self.quantization_step[1], 360.0/self.quantization_step[2]]
                self.voxel_size = [1.0, 1.0, 1.0]
                self.grid_size = np.array([int(XMAX/self.quantization_step[0]),int((YMAX-YMIN)/self.quantization_step[1]),int(360.0/self.quantization_step[2])])
            elif self.coordinates == 'cartesian':
                self.point_cloud_range = [XMIN, ZMIN, YMIN, XMAX, ZMAX, YMAX]
                self.voxel_size = [(XMAX-XMIN)/self.X, (ZMAX-ZMIN)/self.Z, (YMAX-YMIN)/self.Y]
                self.grid_size = np.array([int((XMAX-XMIN)/self.quantization_step[0]),int((ZMAX-ZMIN)/self.quantization_step[0]),int((YMAX-YMIN)/self.quantization_step[2])])
            else:
                raise NotImplementedError('Unknown coordinates: {}'.format(model_params.coordinates))

            # if polar: range (dist, z, theta)
            self.voxel_layer = Voxelization(max_num_points=self.model_params.voxel_num_points,
                                    point_cloud_range=self.point_cloud_range,
                                    voxel_size=self.voxel_size)

            self.vfe_module = vfe.MeanVFE(model_params, 3)

            if 'voxel' in model_params.backbone_3d:
                self.lidar_out_featname = 'encoded_spconv_tensor'
                if self.coordinates == 'polar':
                    self.backbone_3d = backbones_3d.spconv_backbone.VoxelBackBone8x(model_params, 3, self.grid_size)
                elif self.coordinates == 'cartesian':
                    self.backbone_3d = backbones_3d.spconv_backbone.VoxelBackBone8x(model_params, 3, self.grid_size)
            elif 'focal' in model_params.backbone_3d:
                self.lidar_out_featname = 'encoded_spconv_tensor'
                if self.coordinates == 'polar':
                    self.backbone_3d = backbones_3d.spconv_backbone_focal.VoxelBackBone8xFocal(model_params, 3, self.grid_size)
                elif self.coordinates == 'cartesian':
                    self.backbone_3d = backbones_3d.spconv_backbone_focal.VoxelBackBone8xFocal(model_params, 3, self.grid_size)
            elif 'unet' in model_params.backbone_3d:
                self.lidar_out_featname = 'encoded_voxel_features'
                if self.coordinates == 'polar':
                    self.backbone_3d = backbones_3d.spconv_unet.UNetV2(model_params, 3, self.grid_size, self.voxel_size, self.point_cloud_range)
                elif self.coordinates == 'cartesian':
                    pass
            else:
                raise NotImplementedError('Unknown backbone3d: {}'.format(model_params.backbone_3d))

        self.image_meta_path = model_params.image_meta_path
                
        # Image encoder            
        if self.use_rgb:
            self.vox_util = vox.Vox_util(self.Z, self.Y, self.X,
                    scene_centroid=model_params.scene_centroid.cuda(),
                    bounds=model_params.bounds,
                    assert_cube=False)

            self.xyz_memA = basic.gridcloud3d(1, self.Z, self.Y, self.X, norm=False)
            self.xyz_camA = self.vox_util.Mem2Ref(self.xyz_memA, self.Z, self.Y, self.X, assert_cube=False)
        
            if self.use_xyz:
                self.image_feat_dim = self.image_feat_dim // 2

            if model_params.backbone_2d == 'res18':
                self.backbone_2d = Encoder_res18(self.image_feat_dim)
            elif model_params.backbone_2d == 'res50':
                self.backbone_2d = Encoder_res50(self.image_feat_dim)
            elif model_params.backbone_2d == 'unet':
                self.backbone_2d = UNet(3, self.image_feat_dim)
            else:
                raise NotImplementedError('Unknown backbone2d: {}'.format(model_params.backbone_2d))

        # BEV compressor
        self.bev_compressor = nn.Sequential(
            nn.Conv2d(self.feature_dim * self.Y, self.output_dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(self.output_dim ),
            nn.GELU(),
        )

        self.unet = UNet(20,1)
        self.conv = nn.Conv2d(in_channels=self.output_dim, out_channels=1, kernel_size=3, padding=1, stride=1)
        self.fc_out = nn.Linear(self.output_dim, self.output_dim)

        # aggregation
        if self.aggregation == 'gem':
            self.pooling = GeM()
        elif self.aggregation == 'vlad':
            self.pooling = NetVLADLoupe(feature_size=256, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=True)
        elif self.aggregation == 'fft':
            pass
        else:
            raise NotImplementedError('Unknown aggregation method: {}'.format(self.aggregation))


    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                module.bias.data.zero_()

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        # pc: B,V,ndim
        batch_size = points.shape[0]
        coords = []
        voxels = []
        for i in range(batch_size):
            res_voxels, res_coords, num_points_per_voxel = self.voxel_layer(points[i])
            coords.append(res_coords)
            voxel_feat = self.vfe_module(res_voxels, num_points_per_voxel)
            voxels.append(voxel_feat)
        voxel_feat = torch.cat(voxels, dim=0)
        coords_batch = []
        for i, coord in enumerate(coords):
            coord_pad = F.pad(coord, (1, 0), mode='constant', value=i)
            coords_batch.append(coord_pad)
        coords_batch = torch.cat(coords_batch, dim=0)

        return voxel_feat, coords_batch


    def extract_image_feature(self, img):
        # im: Tensor: (B*5, 3, H, W)
        # input_shape = img.shape[-2:]

        img_feats = self.backbone_2d(img)

        _, C, Hf, Wf = img_feats.shape
        sy = Hf/float(self.H)
        sx = Wf/float(self.W)
        return img_feats, sx, sy   # Tensor: (B*5, C, H, W)


    def extract_fused_feature(self, img_feats, lidar_feats):
        # img_feats: Tensor (B, C, Y, radius, theta)
        # lidar_feats: Tensor (B, C, Y, radius, theta)

        ##### simple fusion
        # B = img_feats.shape[0]
        # lidar_voxel_ = lidar_feats.permute(0, 1, 3, 2, 4).reshape(B, self.Y, self.Z, self.X)

        # feat_voxel_ = img_feats.permute(0, 1, 3, 2, 4).reshape(B, self.output_dim*self.Y, self.Z, self.X)
        # feat_voxel_ = torch.cat([feat_voxel_, lidar_voxel_], dim=1)

        ##### deep fusion
        if lidar_feats == None:
            B, C, Y, radius, theta = img_feats.shape
            feat_voxel_ = img_feats.reshape(B, -1, radius, theta)
        elif img_feats == None:
            B, C, Y, radius, theta = lidar_feats.shape
            feat_voxel_ = lidar_feats.reshape(B, -1, radius, theta)
        else:
            B, C, Y, radius, theta = img_feats.shape
            feat_voxel_ = torch.cat([img_feats, lidar_feats], dim=1)
            feat_voxel_ = feat_voxel_.reshape(B, -1, radius, theta)

        #### lidar only
        # feat_voxel_ = img_feats.permute(0, 1, 3, 2, 4).reshape(B, self.Y, self.Z, self.X)
        # feat_voxel_ = torch.cat([feat_voxel_, lidar_voxel_], dim=1)

        return feat_voxel_


    def extract_feature(self, pc, im=None):
        # pc: B,V,4
        # im: Tensor: (B, S, 3, H, W)
        # y_T_x: x in y coordinate
        if pc is not None:
            B, V, D = pc.shape
        else:
            B, S, C, H, W = im.shape
            
        __p = lambda x: basic.pack_seqdim(x, B)
        __u = lambda x: basic.unpack_seqdim(x, B)

        if self.use_rgb:
            _, S, C, H, W = im.shape

            with open(self.image_meta_path, 'rb') as handle:
                image_meta = pickle.load(handle)

            intrins = torch.from_numpy(np.array(image_meta['K'])).float()
            pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
            cams_T_body = torch.from_numpy(np.array(image_meta['T'])).unsqueeze(0).float()

            pix_T_cams = pix_T_cams.repeat(B,1,1,1).cuda()
            cams_T_body = cams_T_body.repeat(B,1,1,1).cuda()

            body_T_cams = __u(geom.safe_inverse(__p(cams_T_body)))
            cam0_T_camXs = __p(geom.get_camM_T_camXs(body_T_cams, ind=0))

            pix_T_cams_ = __p(pix_T_cams)
            camXs_T_cam0_ = geom.safe_inverse(cam0_T_camXs).cuda()

            ###### Image branch
            # reshape tensors
            im = __p(im)

            feat_camXs_, sx, sy = self.extract_image_feature(im)  # (B*S, C, H, W)

            featpix_T_cams_ = geom.scale_intrinsics(pix_T_cams_, sx, sy).cuda()
            xyz_camA = self.xyz_camA.to(feat_camXs_.device).repeat(B*S,1,1)

            feat_mems_ = self.vox_util.unproject_image_to_mem(
                        feat_camXs_,
                        basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
                        camXs_T_cam0_, self.Z, self.Y, self.X,
                        xyz_camA=xyz_camA)

            feat_mems = __u(feat_mems_) # B, S, C, Z, Y, X
            if len(self.use_cam_id) < S:
                feat_mems = feat_mems[:,self.use_cam_id,...]

            mask_mems = (torch.abs(feat_mems) > 0).float()
            feat_mem = basic.reduce_masked_mean(feat_mems, mask_mems, dim=1) # B, C, Z, Y, X
            B, C, Z, Y, X = feat_mem.shape

            ####### Polar transform
            out_h = self.radius
            out_w = self.theta
            new_h = torch.linspace(0, 1, out_h).view(-1, 1).repeat(1, out_w)
            new_w = torch.pi * torch.linspace(0, 2, out_w).repeat(out_h, 1)

            eps = 1e-8
            grid_xy = torch.cat((new_h.unsqueeze(2), new_w.unsqueeze(2)), dim=2)
            new_grid = grid_xy.clone()
            new_grid[...,0] = grid_xy[...,0] * torch.cos(grid_xy[...,1])
            new_grid[...,1] = grid_xy[...,0] * torch.sin(grid_xy[...,1])
            new_grid = new_grid.unsqueeze(0).cuda().repeat(B,1,1,1)
            feat_mem = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, -1, self.Z, self.X)
            polar_img_voxels = F.grid_sample(feat_mem, new_grid, align_corners=False)
            polar_img_voxels = polar_img_voxels.reshape(B, C, Y, self.radius, self.theta)
            del feat_mem
            del new_grid

        else:
            polar_img_voxels = None

        ###### simple Lidar branch
        # xyz_velo0 = pc

        # mag = torch.norm(xyz_velo0[...,:3], dim=2)
        # xyz_velo0 = xyz_velo0[:,mag[0]>1]

        # xyz_cam0 = geom.apply_4x4(cams_T_body[:,0], xyz_velo0[...,:3])
        # xyz_cam0 = pc
        
        # occ_mem0 = self.vox_util.voxelize_xyz(xyz_cam0, self.radius, self.Y, self.theta, assert_cube=False)
        # lidar_voxel_ = occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, self.Y, self.radius, self.theta)
        
        ###### learning Lidar branch
        if self.use_xyz:
            voxel_feat, coords = self.voxelize(pc)   # voxels: voxel_num,ndim; coors: voxel_num,ndim
            batch_dict = {'batch_size': B, 'voxel_features': voxel_feat, 'voxel_coords': coords}
            feat_dict = self.backbone_3d(batch_dict)

        #### Fusion
        if self.coordinates == 'polar':
            # polar_lidar_voxels = feat_dict['encoded_spconv_tensor'].dense()
            if self.use_xyz:
                polar_lidar_voxels = feat_dict[self.lidar_out_featname].dense()
                polar_lidar_voxels = polar_lidar_voxels.permute(0,1,3,4,2)
            else:
                polar_lidar_voxels = None

            fused_voxel_feature = self.extract_fused_feature(polar_img_voxels, polar_lidar_voxels)
            feat_bev_feature = self.bev_compressor(fused_voxel_feature)  # B,C,Radius,Theta

        elif self.coordinates == 'cartesian':
            cart_lidar_voxels = feat_dict[self.lidar_out_featname].dense()
            cart_lidar_voxels = cart_lidar_voxels.reshape(B, -1, self.Z, self.X)

            ####### Polar transform
            polar_lidar_voxels = F.grid_sample(cart_lidar_voxels, new_grid, align_corners=False)
            feat_bev_feature = self.bev_compressor(polar_lidar_voxels)  # B,C,Radius,Theta

        #### simple Fusion

        # fused_voxel_feature = self.extract_fused_feature(feat_mem, occ_mem0)
        # feat_bev_feature = self.bev_compressor(polar_fused_voxels)  # B,C,Radius,Theta

        #### lidar only
        # polar_voxels = F.grid_sample(lidar_voxel_, new_grid, align_corners=False)
        # feat_bev_feature = self.bev_compressor(lidar_voxel_)
        # feat_bev_feature = torch.sum(feat_bev_feature, dim=1, keepdim=True)
        # feat_bev_feature = self.unet(lidar_voxel_)

        return feat_bev_feature


    def forward_fft(self, input):
        median_output = torch.fft.fft2(input, norm="ortho")
        output = torch.sqrt(median_output.real ** 2 + median_output.imag ** 2 + 1e-15)
        output = fftshift2d(output)
        return output, median_output


    def forward(self, batch):
        pc = batch['pc']    # B,V,4 
        im = batch['img']   # B,S,C,H,W; S=cam_num
        
        if im is not None:
            B, S, _, _, _ = im.shape
            assert len(self.use_cam_id) <= S, "camera id set wrong; it's greater than the actual used cameras"

        batch_size = im.shape[0]
        
        x = self.extract_feature(pc, im)
        B, C, rho, theta = x.shape

        if self.aggregation == 'vlad':
            x = x.permute(0,2,3,1)
            x = x.reshape(batch_size, -1, C)
            x = self.pooling(x)
            x = self.fc_out(x)
        elif self.aggregation == 'fft':
            x = self.conv(x)
            x, fourier_spectrum = self.forward_fft(x)
            x = x[:,:, (rho//2 - 8):(rho//2 + 8), (theta//2 - 8):(theta//2 + 8)]
        elif self.aggregation == 'gem':
            x = self.pooling(x)

        x = x.reshape(batch_size, self.output_dim)
    
        if self.use_normalize:
            x = F.normalize(x, dim=1)

        # x is (batch_size, output_dim) tensor
        if self.aggregation == 'fft':
            return {'global': x}
        else:
            return {'global': x}
            

    def print_info(self):
        print('Model class: FusionNet')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))
        n_params = sum([param.nelement() for param in self.bev_compressor.parameters()])
        print('BEV Compressor parameters: {}'.format(n_params))  
        if self.use_rgb:
            n_params = sum([param.nelement() for param in self.backbone_2d.parameters()])
            print('Image Backbone parameters: {}'.format(n_params))
        if self.use_xyz:
            n_params = sum([param.nelement() for param in self.backbone_3d.parameters()])
            print('3D Backbone parameters: {}'.format(n_params))


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def last_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.Sigmoid()
    )


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)



class Decoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        backbone = resnet18(pretrained=False, zero_init_residual=True)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)

        self.feat_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x, bev_flip_indices=None):
        b, c, h, w = x.shape

        # (H, W)
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])

        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            x[bev_flip2_index] = torch.flip(x[bev_flip2_index], [-2]) # note [-2] instead of [-3], since Y is gone now
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])

        feat_output = self.feat_head(x)

        return x



class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class Encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x


class Encoder_res101(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x


class Encoder_res18(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        # for p in self.parameters():
        #     p.requires_grad = False

        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(256, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(384, 256)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x


class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)


class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip

class UNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.dconv_down1 = double_conv(input_dim, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = last_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, output_dim, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


class Encoder_eff(nn.Module):
    def __init__(self, C, version='b4'):
        super().__init__()
        self.C = C
        self.downsample = 8
        self.version = version

        if self.version == 'b0':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        elif self.version == 'b4':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.delete_unused_layers()

        if self.downsample == 16:
            if self.version == 'b0':
                upsampling_in_channels = 320 + 112
            elif self.version == 'b4':
                upsampling_in_channels = 448 + 160
            upsampling_out_channels = 512
        elif self.downsample == 8:
            if self.version == 'b0':
                upsampling_in_channels = 112 + 40
            elif self.version == 'b4':
                upsampling_in_channels = 160 + 56
            upsampling_out_channels = 128
        else:
            raise ValueError(f'Downsample factor {self.downsample} not handled.')

        self.upsampling_layer = UpsamplingConcat(upsampling_in_channels, upsampling_out_channels)
        self.depth_layer = nn.Conv2d(upsampling_out_channels, self.C, kernel_size=1, padding=0)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']
        x = self.upsampling_layer(input_1, input_2)

        return x

    def forward(self, x):
        x = self.get_features(x)  # get feature vector
        x = self.depth_layer(x)  # feature and depth head
        return x
