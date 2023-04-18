import torch
from mmcv.cnn import ConvModule
from torch import nn as nn
from torch.nn import functional as F
import config as cfg


class VoxelFusion(nn.Module):
    def __init__(self):
        super(VoxelFusion, self).__init__()
        self.mlp_q = nn.Linear(in_features=128, out_features=256)
        self.mlp_k = nn.Linear(in_features=256, out_features=256)
        self.mlp_v = nn.Linear(in_features=256, out_features=256)
        self.cross = nn.Linear(in_features=256, out_features=192)
        self.fuse = nn.Linear(in_features=128+192, out_features=128)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                module.bias.data.zero_()

    def cross_attention(self, voxel_pc_features, voxel_im_features):
        # voxel_corrs (p*B, 4)
        # voxel_pc_features (p*B, 128)
        # voxel_im_features (p*B, N, 256)
        q = self.mlp_q(voxel_pc_features)
        k = self.mlp_k(voxel_im_features)
        v = self.mlp_v(voxel_im_features)
        similarity = torch.matmul(k, q.unsqueeze(dim=-1))  # (p*B, N, 1)
        similarity = F.softmax(similarity, dim=1)
        res = (similarity * v).sum(dim=1)
        cam_feature = self.cross(res)
        fused_feature = self.fuse(torch.cat([cam_feature, voxel_pc_features], dim=1))
        return fused_feature

    def forward(self, voxel_features, voxel_coors, voxel_means, image_feature, image_meta):
        # voxel_features (p*B, 128) if point fusion else (p*B, 64)
        # voxel_coors (p*B, 4)
        # voxel_mean (p*B, 4)
        device = voxel_means.device
        batch_size = voxel_coors[-1, 0] + 1
        voxel_means[:, 3] = torch.round(voxel_means[:, 3])
        voxel_coors_, voxel_cam_features_, voxel_pc_features_ = torch.Tensor([]).to(
            device), torch.Tensor([]).to(device), torch.Tensor([]).to(device)
        for batch_id in range(batch_size):
            for i in range(6):
                mask = (voxel_means[:, 3] == i) & (voxel_coors[:, 0] == batch_id)
                voxel_coors_ = torch.cat([voxel_coors_, voxel_coors[mask]], dim=0)
                voxel_pc_features_ = torch.cat([voxel_pc_features_, voxel_features[mask]], dim=0)
                points_camera = voxel_means[mask, 0:3]
                ones = torch.ones((points_camera.shape[0], 1)).to(device)
                points_camera = torch.cat([points_camera, ones], dim=1)
                T = torch.from_numpy(image_meta["T"][i]).float().to(device)
                K = torch.from_numpy(image_meta["K"][i]).float().to(device)
                K[0][0] /= 8
                K[0][2] /= 8
                K[1][1] /= 8
                K[1][2] /= 8
                points_c = torch.matmul(T, points_camera.t())
                points_im = torch.matmul(K, points_c[:3, :]).t()
                points_im = (points_im / points_im[:, 2:3])[:, :2]  # (p, 2) -> (p, 11, 2) -> (p, 11, 256)
                P = points_im.shape[0]
                offset = torch.Tensor([[-1, 0], [-0.8, 0], [-0.6, 0], [-0.4, 0], [-0.2, 0], [0, 0],
                                       [0.2, 0], [0.4, 0], [0.6, 0], [0.8, 0], [1, 0]]).to(device)
                points_im_index = torch.stack([points_im + offset[i:i + 1]
                                               for i in range(offset.shape[0])], dim=1).reshape(-1, 2).clamp(0, cfg.IMAGE_WIDTH / 8)
                coor_x, coor_y = torch.split(points_im_index, 1, dim=1)
                h, w = cfg.IMAGE_HEIGHT / 8, cfg.IMAGE_WIDTH / 8
                coor_y = coor_y / h * 2 - 1
                coor_x = coor_x / w * 2 - 1
                grid = torch.cat([coor_x, coor_y], dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2
                point_features = F.grid_sample(
                    image_feature[0][i:i+1, batch_id, ...],
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False)  # 1xCx1xN feats
                voxel_cam_features_ = torch.cat([voxel_cam_features_, point_features.squeeze(0).squeeze(1).t().reshape(P, offset.shape[0], 256)], dim=0)
        # cross attention
        fused_feature = self.cross_attention(voxel_pc_features_, voxel_cam_features_)
        return fused_feature
