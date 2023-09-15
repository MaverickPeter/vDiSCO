# Zhejiang University

import argparse
import numpy as np
import tqdm
import os
import random
from typing import List
import open3d as o3d
from time import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
import torch.nn as nn
from torchvision import transforms as transforms
from prnet.utils.data_utils.poses import m2ypr, relative_pose
from prnet.datasets.nclt.utils import relative_pose as nclt_relative_pose
from prnet.utils.data_utils.point_clouds import icp, make_open3d_feature, make_open3d_point_cloud
from prnet.models.model_factory import model_factory
from prnet.utils.data_utils.poses import apply_transform, m2ypr
from prnet.utils.params import TrainingParams, ModelParams
from prnet.datasets.dataset_utils import preprocess_pointcloud
from prnet.datasets.base_datasets import EvaluationTuple, EvaluationSet, get_pointcloud_loader, get_pointcloud_with_image_loader
from prnet.datasets.panorama import generate_sph_image
from prnet.utils.loss_utils import *
from prnet.datasets.range_image import range_projection


class Evaluator:
    def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str, params: ModelParams,
                 radius: List[float] = [1.5, 5, 20], k: int = 50, n_samples: int =None, debug: bool = False):
        # radius: list of thresholds (in meters) to consider an element from the map sequence a true positive
        # k: maximum number of nearest neighbours to consider
        # n_samples: number of samples taken from a query sequence (None=all query elements)

        assert os.path.exists(dataset_root), f"Cannot access dataset root: {dataset_root}"
        self.dataset_root = dataset_root
        self.dataset_type = dataset_type
        self.eval_set_filepath = os.path.join(dataset_root, eval_set_pickle)
        self.device = device
        self.radius = radius
        self.k = k
        self.n_samples = n_samples
        self.debug = debug
        self.params = params

        assert os.path.exists(self.eval_set_filepath), f'Cannot access evaluation set pickle: {self.eval_set_filepath}'
        self.eval_set = EvaluationSet()
        self.eval_set.load(self.eval_set_filepath)
        if debug:
            # Make the same map set and query set in debug mdoe
            self.eval_set.map_set = self.eval_set.map_set[:4]
            self.eval_set.query_set = self.eval_set.map_set[:4]

        if n_samples is None or len(self.eval_set.query_set) <= n_samples:
            self.n_samples = len(self.eval_set.query_set)
        else:
            self.n_samples = n_samples

        if self.params.use_rgb:
            self.pcim_loader = get_pointcloud_with_image_loader(self.dataset_type)
        else:
            self.pcim_loader = get_pointcloud_loader(self.dataset_type)

    def evaluate(self, model, *args, **kwargs):
        map_embeddings = self.compute_embeddings(self.eval_set.map_set, model)
        query_embeddings = self.compute_embeddings(self.eval_set.query_set, model)

        map_positions = self.eval_set.get_map_positions()
        query_positions = self.eval_set.get_query_positions()

        # Dictionary to store the number of true positives for different radius and NN number
        tp = {r: [0] * self.k for r in self.radius}
        query_indexes = random.sample(range(len(query_embeddings)), self.n_samples)

        # Randomly sample n_samples clouds from the query sequence and NN search in the target sequence
        for query_ndx in tqdm.tqdm(query_indexes):
            # Check if the query element has a true match within each radius
            query_pos = query_positions[query_ndx]

            # Nearest neighbour search in the embedding space
            query_embedding = query_embeddings[query_ndx]
            embed_dist = np.linalg.norm(map_embeddings - query_embedding, axis=1)
            nn_ndx = np.argsort(embed_dist)[:self.k]

            # Euclidean distance between the query and nn
            delta = query_pos - map_positions[nn_ndx]  # (k, 2) array
            euclid_dist = np.linalg.norm(delta, axis=1)  # (k,) array
            # Count true positives for different radius and NN number
            tp = {r: [tp[r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in
                  self.radius}

        recall = {r: [tp[r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}
        # percentage of 'positive' queries (with at least one match in the map sequence within given radius)
        return {'recall': recall}

    def compute_embedding(self, pc, depth, imgs, model, *args, **kwargs):
        # This method must be implemented in inheriting classes
        # Must return embedding as a numpy vector
        raise NotImplementedError('Not implemented')

    def model2eval(self, model):
        # This method may be overloaded when model is a tuple consisting of a few models (as in Disco)
        model.eval()

    def compute_embeddings(self, eval_subset: List[EvaluationTuple], model, *args, **kwargs):
        self.model2eval(model)

        embeddings = None
        for ndx, e in tqdm.tqdm(enumerate(eval_subset)):
            scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)
            assert os.path.exists(scan_filepath)
            pc, imgs = self.pcim_loader(scan_filepath)
            pc = torch.tensor(pc)

            embedding = self.compute_embedding(pc, imgs, model)
            if embeddings is None:
                embeddings = np.zeros((len(eval_subset), embedding.shape[1]), dtype=embedding.dtype)
            embeddings[ndx] = embedding

        return embeddings


class GLEvaluator(Evaluator):
    # Evaluation of EgoNN methods on Mulan or Apollo SouthBay dataset
    def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str, params: ModelParams,
                 radius: List[float], k: int = 20, n_samples=None, debug: bool = False):
        super().__init__(dataset_root, dataset_type, eval_set_pickle, device, params, radius, k, n_samples, debug=debug)
        self.params = params

    def model2eval(self, models):
        # This method may be overloaded when model is a tuple consisting of a few models (as in Disco)
        [model.eval() for model in models]

    def evaluate(self, model, *args, **kwargs):
        map_embeddings, map_specs, eval_yaw = self.compute_embeddings(self.eval_set.map_set, model)
        query_embeddings, query_specs, _ = self.compute_embeddings(self.eval_set.query_set, model)
        if eval_yaw:
            map_specs = torch.from_numpy(map_specs)
            query_specs = torch.from_numpy(query_specs)

        map_positions = self.eval_set.get_map_positions()
        query_positions = self.eval_set.get_query_positions()
        map_poses = self.eval_set.get_map_poses()
        query_poses = self.eval_set.get_query_poses()

        if self.n_samples is None or len(query_embeddings) <= self.n_samples:
            query_indexes = list(range(len(query_embeddings)))
            self.n_samples = len(query_embeddings)
        else:
            query_indexes = random.sample(range(len(query_embeddings)), self.n_samples)

        metrics = {}

        # Dictionary to store the number of true positives (for global desc. metrics) for different radius and NN number
        global_metrics = {'tp': {r: [0] * self.k for r in self.radius}}

        error_metrics = []

        for query_ndx in tqdm.tqdm(query_indexes):
            # Check if the query element has a true match within each radius
            query_pos = query_positions[query_ndx]
            query_pose = self.eval_set.query_set[query_ndx].pose
            qyaw, qpitch, qroll = m2ypr(query_pose)

            # Nearest neighbour search in the embedding space
            query_embedding = query_embeddings[query_ndx]
            embed_dist = np.linalg.norm(map_embeddings - query_embedding, axis=1)
            nn_ndx = np.argsort(embed_dist)[:self.k]

            # GLOBAL DESCRIPTOR EVALUATION
            # Euclidean distance between the query and nn
            # Here we use non-icp refined poses, but for the global descriptor it's fine
            delta = query_pos - map_positions[nn_ndx]       # (k, 2) array

            if eval_yaw:
                query_spec = query_specs[query_ndx]
                map_pose = self.eval_set.map_set[nn_ndx[0]].pose
                myaw, mpitch, mroll = m2ypr(map_pose)
                _, corr = phase_corr(query_spec.unsqueeze(0), map_specs[nn_ndx[0]].unsqueeze(0))
                angle = torch.argmax(corr).float()
                angle -= (self.params.theta / 2.)
                pred_angle = angle / self.params.theta * 360.

                gt_yaw_diff = (qyaw - myaw) / np.pi * 180.
                error = np.min([np.abs(gt_yaw_diff - pred_angle), np.abs(np.abs((gt_yaw_diff - pred_angle))-360.)])
                error_metrics.append(error)

            euclid_dist = np.linalg.norm(delta, axis=1)     # (k,) array

            global_metrics['tp'] = {r: [global_metrics['tp'][r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in self.radius}
            

        if eval_yaw:
            error_metrics = np.stack(error_metrics)
            yaw_err_std = 0.
            yaw_err_mean = np.mean(error_metrics)
            yaw_err_median = np.median(error_metrics)

            for ndx in query_indexes:
                yaw_err_std += np.power((error_metrics[ndx] - yaw_err_mean), 2)

            yaw_err_std = np.sqrt(yaw_err_std / len(query_indexes))
            global_metrics["yaw_err_mean"] = yaw_err_mean
            global_metrics["yaw_err_median"] = yaw_err_median
            global_metrics["yaw_err_std"] = yaw_err_std

            print("25%: ", np.quantile(error_metrics, 0.25))
            print("50%: ", np.quantile(error_metrics, 0.50))
            print("75%: ", np.quantile(error_metrics, 0.75))


        # Calculate mean metrics
        global_metrics["recall"] = {r: [global_metrics['tp'][r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}

        return global_metrics


    def compute_embeddings(self, eval_subset: List[EvaluationTuple], model, *args, **kwargs):
        self.model2eval((model,))
        global_embeddings = None
        specs = None
        depths = None

        for ndx, e in tqdm.tqdm(enumerate(eval_subset)):
            if self.params.use_panorama:
                sph = True
            else:
                sph = False

            if self.params.dataset_type == 'oxford':
                extrinsics_dir = os.path.join(self.params.dataset_folder, 'extrinsics')
                pc, imgs = self.pcim_loader(e.filepaths, sph, extrinsics_dir)
            elif self.params.dataset_type == 'nclt':
                scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)
                assert os.path.exists(scan_filepath)
                pc, imgs = self.pcim_loader(scan_filepath, sph)


            if self.params.use_range_image:
                depths = np.zeros((1,64,900), dtype=np.float32)
                cloud = np.ones((pc.shape[0],4))
                cloud[...,:3] = pc[...,:3]
                depth, _, _, _ = range_projection(cloud)
                depths[0,...] = depth
                depths = torch.from_numpy(depths).cuda()
            else:
                V = self.params.lidar_fix_num
                lidar_data = pc[:,:3]
                lidar_extra = pc[:,3:]
                if pc.shape[0] > V:
                    lidar_data = lidar_data[:V]
                    lidar_extra = lidar_extra[:V]
                elif pc.shape[0] < V:
                    lidar_data = np.pad(lidar_data,[(0,V-lidar_data.shape[0]),(0,0)],mode='constant')
                    lidar_extra = np.pad(lidar_extra,[(0,V-lidar_extra.shape[0]),(0,0)],mode='constant',constant_values=-1)
                
                lidar_data = torch.from_numpy(lidar_data).float()
                lidar_extra = torch.from_numpy(lidar_extra).float()

                pc = torch.cat([lidar_data, lidar_extra], dim=1)   
                pc = pc.unsqueeze(0)

                if self.params.quantizer == None:
                    pc = pc
                else:
                    pc = self.params.quantizer(pc)
                pc = pc.cuda()


            toTensor = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            if self.params.use_rgb:
                imgs = [toTensor(e) for e in imgs]
                imgs = torch.stack(imgs).float().cuda()
                if not sph:
                    imgs = imgs.unsqueeze(0).cuda()
            else:
                imgs = None

            global_embedding, spec = self.compute_embedding(pc, depths, imgs, model)

            if global_embeddings is None:
                global_embeddings = np.zeros((len(eval_subset), global_embedding.shape[1]), dtype=global_embedding.dtype)

            if spec is not None:
                if specs is None:
                    specs = np.zeros((len(eval_subset), spec.shape[-3], spec.shape[-2], spec.shape[-1]), dtype=spec.dtype)

                specs[ndx] = spec
                eval_yaw = True
            else:
                eval_yaw = False

            global_embeddings[ndx] = global_embedding

        return global_embeddings, specs, eval_yaw

    def compute_embedding(self, pc, depth, imgs, model, *args, **kwargs):
        """
        Returns global embedding (np.array)
        """
        with torch.no_grad():
            if self.params.use_range_image:
                batch = {'depth': depth, 'img': imgs}
            else:
                batch = {'pc': pc, 'img': imgs}
            
            # Compute global descriptor
            y = model(batch)

            global_embedding = y['global'].detach().cpu().numpy()
            if 'spectrum' in y.keys():
                spec = y['spectrum'].detach().cpu().numpy()
            else:
                spec = None

        return global_embedding, spec


    def print_results(self, global_metrics):
        # Global descriptor results are saved
        recall = global_metrics['recall']
        for r in recall:
            print(f"Radius: {r} [m] : ", end='')
            for x in recall[r]:
                print("{:0.3f}, ".format(x), end='')
            print("")
            
        if 'yaw_err_mean' in global_metrics.keys():
            yaw_err_mean = global_metrics['yaw_err_mean']
            print(f"Yaw err mean [deg] : ", end='')
            print("{:0.3f}, ".format(yaw_err_mean), end='')
            print("")
            
        if 'yaw_err_std' in global_metrics.keys():
            yaw_err_std = global_metrics['yaw_err_std']
            print(f"Yaw err std [deg] : ", end='')
            print("{:0.3f}, ".format(yaw_err_std), end='')
            print("")

        if 'yaw_err_median' in global_metrics.keys():
            yaw_err_median = global_metrics['yaw_err_median']
            print(f"Yaw err median [deg] : ", end='')
            print("{:0.3f}, ".format(yaw_err_median), end='')
            print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Fusion model')
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to the dataset root')
    parser.add_argument('--dataset_type', type=str, required=True, choices=['mulran', 'southbay', 'kitti', 'nclt','oxford'])
    parser.add_argument('--eval_set', type=str, required=True, help='File name of the evaluation pickle (must be located in dataset_root')
    parser.add_argument('--radius', type=float, nargs='+', default=[2, 5, 10, 20], help='True Positive thresholds in meters')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of elements sampled from the query sequence')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the global model configuration file')
    parser.add_argument('--weights', type=str, default=None, help='Trained global model weights')

    args = parser.parse_args()
    print(f'Dataset root: {args.dataset_root}')
    print(f'Dataset type: {args.dataset_type}')
    print(f'Evaluation set: {args.eval_set}')
    print(f'Radius: {args.radius} [m]')
    print(f'Number of sampled query elements: {args.n_samples}')
    print(f'Model config path: {args.model_config}')
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print(f'Weights: {w}')
    print('')

    model_params = ModelParams(args.model_config, args.dataset_type, args.dataset_root)
    model_params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(model_params)
    corr2soft = Corr2Softmax(200., 0.)
    corr2soft = corr2soft.to(device)

    if args.weights is not None:
        assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
        print('Loading weights: {}'.format(args.weights))
        
        # state_dict = torch.load(args.weights, map_location=device)
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v
        # model.load_state_dict(new_state_dict, strict=True)
        
        model = nn.DataParallel(model)
        if 'netvlad' in model_params.model and 'pretrain' in model_params.model and 'finetune' not in model_params.model:
            checkpoint = torch.load(args.weights, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            checkpoint = torch.load(args.weights, map_location=device)
            model.load_state_dict(checkpoint, strict=True)
            # model.load_state_dict(checkpoint['model'], strict=True)
            # corr2soft.load_state_dict(checkpoint['corr2soft'], strict=True)

    model.to(device)
    
    evaluator = GLEvaluator(args.dataset_root, args.dataset_type, args.eval_set, device, radius=args.radius,
                                n_samples=args.n_samples, params=model_params)
    global_metrics = evaluator.evaluate(model)
    evaluator.print_results(global_metrics)
