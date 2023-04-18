import torch
from mmcv.cnn import ConvModule
from torch import nn as nn
from torch.nn import functional as F
import config as cfg
cfg.IMAGE_HEIGHT = 360
cfg.IMAGE_WIDTH = 240

def point_sample(img_meta,
                 img_features,
                 points,
                 aligned=True,
                 padding_mode='zeros',
                 align_corners=True):
    points_feature = []
    for i in range(5):
        mask = points[:, 3] == i
        points_camera = points[mask, 0:3]
        device = points.device
        ones = torch.ones((points_camera.shape[0], 1)).to(device)
        points_camera = torch.cat([points_camera, ones], dim=1)
        T = torch.from_numpy(img_meta["T"][i]).float().to(device)
        K = torch.from_numpy(img_meta["K"][i]).float().to(device)
        K[0][0] /= 8
        K[0][2] /= 8
        K[1][1] /= 8
        K[1][2] /= 8
        points_c = torch.matmul(T, points_camera.t())
        points_im = torch.matmul(K, points_c[:3, :]).t()
        points_im = (points_im / points_im[:, 2:3])[:, :2]

        # grid sample, the valid grid range should be in [-1,1]
        coor_x, coor_y = torch.split(points_im, 1, dim=1)  # each is Nx1
        h, w = cfg.IMAGE_HEIGHT / 8, cfg.IMAGE_WIDTH / 8
        coor_y = coor_y / h * 2 - 1
        coor_x = coor_x / w * 2 - 1
        grid = torch.cat([coor_x, coor_y],
                         dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

        mode = 'bilinear' if aligned else 'nearest'
        point_features = F.grid_sample(
            img_features[:, i, ...],
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners)  # 1xCx1xN feats
        points_feature.append(point_features.squeeze(0).squeeze(1).t())
    return torch.cat(points_feature, dim=0)


class PointFusion(nn.Module):
    def __init__(self,
                 img_channels,
                 pts_channels,
                 mid_channels,
                 out_channels,
                 img_levels=3,
                 coord_type='LIDAR',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None,
                 activate_out=True,
                 fuse_out=False,
                 dropout_ratio=0,
                 aligned=True,
                 align_corners=True,
                 padding_mode='zeros',
                 lateral_conv=True):
        super(PointFusion, self).__init__()
        if isinstance(img_levels, int):
            img_levels = [img_levels]
        if isinstance(img_channels, int):
            img_channels = [img_channels] * len(img_levels)
        assert isinstance(img_levels, list)
        assert isinstance(img_channels, list)
        assert len(img_channels) == len(img_levels)

        self.img_levels = img_levels
        self.coord_type = coord_type
        self.act_cfg = act_cfg
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        self.dropout_ratio = dropout_ratio
        self.img_channels = img_channels
        self.aligned = aligned
        self.align_corners = align_corners
        self.padding_mode = padding_mode

        self.lateral_convs = None
        if lateral_conv:
            self.lateral_convs = nn.ModuleList()
            for i in range(len(img_channels)):
                l_conv = ConvModule(
                    img_channels[i],
                    mid_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                self.lateral_convs.append(l_conv)
            self.img_transform = nn.Sequential(
                nn.Linear(mid_channels * len(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        else:
            self.img_transform = nn.Sequential(
                nn.Linear(sum(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        self.pts_transform = nn.Sequential(
            nn.Linear(pts_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        )

        if self.fuse_out:
            self.fuse_conv = nn.Sequential(
                nn.Linear(mid_channels, out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=False))

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                dict(type='Xavier', layer='Linear', distribution='uniform')
            ]

    def forward(self, img_feats, pts, pts_feats, img_metas):
        # img_feats: Tensor (B*5, C, H, W)
        # pts: [n*4, n*4, ...] B lidar points
        img_pts = self.obtain_mlvl_feats(img_feats, pts, img_metas)
        img_pre_fuse = self.img_transform(img_pts)
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio)
        pts_pre_fuse = self.pts_transform(pts_feats)

        fuse_out = img_pre_fuse + pts_pre_fuse
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)

        return fuse_out

    def obtain_mlvl_feats(self, img_feats, pts, img_metas):
        if self.lateral_convs is not None:
            img_inses = []
            for k, lateral_conv in zip(self.img_levels, self.lateral_convs):
                img_ins = lateral_conv(img_feats[k])
                BN, C, H, W = img_ins.size()
                img_ins = img_ins.reshape(int(BN/5), 5, C, H, W)
                img_inses.append(img_ins)
        else:
            img_inses = img_feats
        img_feats_per_point = []
        for i in range(len(pts)):
            mlvl_img_feats = []
            for level in range(len(self.img_levels)):
                mlvl_img_feats.append(
                    point_sample(img_features=img_inses[level][i:i+1, ...],
                                 points=pts[i],
                                 img_meta=img_metas,
                                 aligned=self.aligned,
                                 padding_mode=self.padding_mode,
                                 align_corners=self.align_corners
                                 ))
            mlvl_img_feats = torch.cat(mlvl_img_feats, dim=-1)
            img_feats_per_point.append(mlvl_img_feats)

        img_pts = torch.cat(img_feats_per_point, dim=0)
        return img_pts
