# Base dataset classes, inherited by dataset-specific classes
import os
import pickle
from typing import List, Dict
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as transforms

from prnet.datasets.nclt.nclt_raw import NCLTPointCloudLoader, NCLTPointCloudWithImageLoader
from prnet.datasets.oxford.oxford_raw import OxfordPointCloudLoader, OxfordPointCloudWithImageLoader
from prnet.datasets.kitti360.kitti360_raw import Kitti360PointCloudLoader, Kitti360PointCloudWithImageLoader
from prnet.utils.data_utils.point_clouds import PointCloudLoader, PointCloudWithImageLoader
from prnet.utils.data_utils.poses import m2ypr
from prnet.datasets.panorama import generate_sph_image


class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, pose: np, positives_poses: Dict[int, np.ndarray] = None, filepaths: list = None):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # pose: pose as 4x4 matrix
        # positives_poses: relative poses of positive examples refined using ICP
        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.pose = pose
        self.positives_poses = positives_poses
        self.filepaths = filepaths


class EvaluationTuple:
    # Tuple describing an evaluation set element
    def __init__(self, timestamp: int, rel_scan_filepath: str, position: np.array, pose: np.array = None, filepaths: list = None):
        # position: x, y position in meters
        # pose: 6 DoF pose (as 4x4 pose matrix)
        assert position.shape == (2,)
        assert pose is None or pose.shape == (4, 4)
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.position = position
        self.pose = pose
        self.filepaths = filepaths

    def to_tuple(self):
        return self.timestamp, self.rel_scan_filepath, self.position, self.pose, self.filepaths


class TrainingDataset(Dataset):
    def __init__(self, dataset_path: str, dataset_type: str, query_filename: str, transform=None, set_transform=None, image_transform=None, params=None):
        # remove_zero_points: remove points with all zero coords
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        print("dataset type ", dataset_type)
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform
        self.image_transform = image_transform
        self.queries: Dict[int, TrainingTuple] = pickle.load(open(self.query_filepath, 'rb'))
        print('{} queries in the dataset'.format(len(self)))

        self.params = params
        if self.dataset_type == 'oxford':
            self.extrinsics_dir = os.path.join(self.dataset_path, 'extrinsics')
        else:
            self.extrinsics_dir = None

        # pc_loader must be set in the inheriting class
        if self.params.model_params.use_rgb:
            self.pcim_loader = get_pointcloud_with_image_loader(self.dataset_type)
        else:
            self.pcim_loader = get_pointcloud_loader(self.dataset_type)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        if self.params.model_params.use_panorama:
            sph = True
        else:
            sph = False

        if self.dataset_type == 'oxford':
            query_pc, query_imgs = self.pcim_loader(self.queries[ndx].filepaths, sph, self.extrinsics_dir)
        else:
            file_pathname = os.path.join(self.dataset_path, self.queries[ndx].rel_scan_filepath)
            query_pc, query_imgs = self.pcim_loader(file_pathname, sph)
            if self.params.use_overlap:
                query_depth_path = file_pathname.replace('velodyne_sync', 'depth')
                query_depth = np.load(query_depth_path.replace('bin','npy'))

        # ground truth yaw
        query_yaw, pitch, roll = m2ypr(self.queries[ndx].pose)
        query_pc = torch.tensor(query_pc, dtype=torch.float)

        if self.transform is not None:
            query_pc = self.transform(query_pc)

        if self.params.model_params.use_rgb:
            if self.image_transform is not None:
                query_imgs = [self.image_transform(e) for e in query_imgs]
            else:
                t = [transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                transform = transforms.Compose(t)
                query_imgs = [transform(e) for e in query_imgs]

            query_imgs = torch.stack(query_imgs).float()

            if self.params.use_overlap:
                query_depth = [torch.from_numpy(e) for e in query_depth]
                query_depth = torch.stack(query_depth).float()
                return query_pc, query_imgs, self.queries[ndx].pose, ndx, query_yaw
        
        return query_pc, query_imgs, ndx, query_yaw


    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives


class EvaluationSet:
    # Evaluation set consisting of map and query elements
    def __init__(self, query_set: List[EvaluationTuple] = None, map_set: List[EvaluationTuple] = None):
        self.query_set = query_set
        self.map_set = map_set

    def save(self, pickle_filepath: str):
        # Pickle the evaluation set

        # Convert data to tuples and save as tuples
        query_l = []
        for e in self.query_set:
            query_l.append(e.to_tuple())

        map_l = []
        for e in self.map_set:
            map_l.append(e.to_tuple())
        pickle.dump([query_l, map_l], open(pickle_filepath, 'wb'))

    def load(self, pickle_filepath: str):
        # Load evaluation set from the pickle
        query_l, map_l = pickle.load(open(pickle_filepath, 'rb'))

        self.query_set = []
        for e in query_l:
            if len(e) == 5:
                self.query_set.append(EvaluationTuple(e[0], e[1], e[2], e[3], e[4]))
            else:
                self.query_set.append(EvaluationTuple(e[0], e[1], e[2], e[3], None))

        self.map_set = []
        for e in map_l:
            if len(e) == 5:
                self.map_set.append(EvaluationTuple(e[0], e[1], e[2], e[3], e[4]))
            else:
                self.map_set.append(EvaluationTuple(e[0], e[1], e[2], e[3], None))


    def get_map_positions(self):
        # Get map positions as (N, 2) array
        positions = np.zeros((len(self.map_set), 2), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            positions[ndx] = pos.position
        return positions

    def get_query_positions(self):
        # Get query positions as (N, 2) array
        positions = np.zeros((len(self.query_set), 2), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            positions[ndx] = pos.position
        return positions

    def get_map_poses(self):
        # Get map positions as (N, 2) array
        poses = np.zeros((len(self.map_set), 4, 4), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            poses[ndx] = pos.pose
        return poses

    def get_query_poses(self):
        # Get query positions as (N, 2) array
        poses = np.zeros((len(self.query_set), 4, 4), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            poses[ndx] = pos.pose
        return poses

def get_pointcloud_loader(dataset_type) -> PointCloudLoader:
    if dataset_type == 'nclt':
        return NCLTPointCloudLoader()
    elif dataset_type == 'oxford':
        return OxfordPointCloudLoader()
    elif dataset_type == 'kitti360':
        return Kitti360PointCloudLoader()
    else:
        raise NotImplementedError(f"Unsupported dataset type: {dataset_type}")


def get_pointcloud_with_image_loader(dataset_type) -> PointCloudWithImageLoader:
    if dataset_type == 'nclt':
        return NCLTPointCloudWithImageLoader()
    elif dataset_type == 'oxford':
        return OxfordPointCloudWithImageLoader()
    elif dataset_type == 'kitti360':
        return Kitti360PointCloudWithImageLoader()
    else:
        raise NotImplementedError(f"Unsupported dataset type: {dataset_type}")


        