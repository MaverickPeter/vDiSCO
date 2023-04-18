# Training tuples generation for NCLT dataset.

import numpy as np
import argparse
import tqdm
import pickle
import os
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import open3d as o3d
from prnet.datasets.nclt.nclt_raw import *
from prnet.datasets.base_datasets import TrainingTuple
from prnet.datasets.nclt.utils import relative_pose
from prnet.utils.data_utils.point_clouds import icp
from prnet.datasets.panorama import generate_sph_image
from prnet.utils.data_utils.poses import m2ypr

DEBUG = True
ICP_REFINE = False


def load_pc(file_pathname):
    # Load point cloud, clip x, y and z coords (points far away and the ground plane)
    # Returns Nx3 matrix
    pc = load_lidar_file_nclt(file_pathname)

    mask = np.all(np.isclose(pc, 0.), axis=1)
    pc = pc[~mask]
    mask = pc[:, 0] > -80
    pc = pc[mask]
    mask = pc[:, 0] <= 80

    pc = pc[mask]
    mask = pc[:, 1] > -80
    pc = pc[mask]
    mask = pc[:, 1] <= 80
    pc = pc[mask]

    mask = pc[:, 2] > 0.0
    pc = pc[mask]


    return pc

def pc2image_file(pc_filename, vel_folder, cam_num, vel_type):
    img_filename = pc_filename.replace(vel_type, '.jpg')
    img_filename = img_filename.replace(vel_folder, '/lb3_u_s_384/Cam' + str(cam_num) + "/")
    # print(img_filename)
    return img_filename


def load_im_file_for_generate(filename):
    input_image = cv2.imread(filename)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.rotate(input_image, cv2.ROTATE_90_CLOCKWISE) #ROTATE_90_COUNTERCLOCKWISE
    return input_image


def generate_training_tuples(ds: NCLTSequences, pos_threshold: float = 10, neg_threshold: float = 50):
    # displacement: displacement between consecutive anchors (if None all scans are takes as anchors).
    #               Use some small displacement to ensure there's only one scan if the vehicle does not move

    tuples = {}   # Dictionary of training tuples: tuples[ndx] = (sef ot positives, set of non negatives)
    for anchor_ndx in tqdm.tqdm(range(len(ds))):
        # reading_filepath = os.path.join(ds.dataset_root, ds.rel_scan_filepath[anchor_ndx])
        # images = [load_im_file_for_generate(pc2image_file(reading_filepath, '/velodyne_sync/', i, '.bin')) for i in range(1, 6)]
        # sph_filename = reading_filepath.replace('bin', 'jpg')
        # sph_filename = sph_filename.replace('velodyne_sync', 'sph')
        # sph_img = generate_sph_image(images, 'nclt', ds.dataset_root)
        # # print(sph_filename)
        # cv2.imwrite(sph_filename, sph_img)
        # query_yaw, pitch, roll = m2ypr(ds.poses[anchor_ndx])

        anchor_pos = ds.get_xy()[anchor_ndx]

        # Find timestamps of positive and negative elements
        positives = ds.find_neighbours_ndx(anchor_pos, pos_threshold)
        non_negatives = ds.find_neighbours_ndx(anchor_pos, neg_threshold)
        # Remove anchor element from positives, but leave it in non_negatives
        positives = positives[positives != anchor_ndx]

        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        # db_yaw, pitch, roll = m2ypr(ds.poses[positives[5]])
        # if (db_yaw - query_yaw) /np.pi * 180. > 30.0:
        #     reading_filepath = os.path.join(ds.dataset_root, ds.rel_scan_filepath[positives[5]])
        #     images = [load_im_file_for_generate(pc2image_file(reading_filepath, '/velodyne_sync/', i, '.bin')) for i in range(1, 6)]
        #     sph_filename = reading_filepath.replace('bin', 'jpg')
        #     sph_filename = sph_filename.replace('velodyne_sync', 'sph')
        #     sph_img = generate_sph_image(images, 'nclt', ds.dataset_root)
        #     # cv2.imwrite(sph_filename, sph_img)
        #     cv2.imwrite("/mnt/workspace/db.png", sph_img)

        # ICP pose refinement
        fitness_l = []
        inlier_rmse_l = []
        positive_poses = {}

        if DEBUG:
            # Use ground truth transform without pose refinement
            anchor_pose = ds.poses[anchor_ndx]
            for positive_ndx in positives:
                positive_pose = ds.poses[positive_ndx]
                # Compute initial relative pose
                m, fitness, inlier_rmse = relative_pose(anchor_pose, positive_pose), 1., 1.
                fitness_l.append(fitness)
                inlier_rmse_l.append(inlier_rmse)
                positive_poses[positive_ndx] = m
        else:
            anchor_pc = load_pc(os.path.join(ds.dataset_root, ds.rel_scan_filepath[anchor_ndx]))
            anchor_pose = ds.poses[anchor_ndx]
            for positive_ndx in positives:  
                positive_pose = ds.poses[positive_ndx]
                transform = relative_pose(anchor_pose, positive_pose)
                if ICP_REFINE:
                    positive_pc = load_pc(os.path.join(ds.dataset_root, ds.rel_scan_filepath[positive_ndx]))
                    # Compute initial relative pose
                    # Refine the pose using ICP
                    m, fitness, inlier_rmse = icp(anchor_pc, positive_pc, transform)

                    fitness_l.append(fitness)
                    inlier_rmse_l.append(inlier_rmse)
                    positive_poses[positive_ndx] = m
                positive_poses[positive_ndx] = transform

        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        tuples[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=ds.timestamps[anchor_ndx],
                                           rel_scan_filepath=ds.rel_scan_filepath[anchor_ndx],
                                           positives=positives, non_negatives=non_negatives, pose=anchor_pose,
                                           positives_poses=positive_poses)

    print(f'{len(tuples)} training tuples generated')
    if ICP_REFINE:
        print('ICP pose refimenement stats:')
        print(f'Fitness - min: {np.min(fitness_l):0.3f}   mean: {np.mean(fitness_l):0.3f}   max: {np.max(fitness_l):0.3f}')
        print(f'Inlier RMSE - min: {np.min(inlier_rmse_l):0.3f}   mean: {np.mean(inlier_rmse_l):0.3f}   max: {np.max(inlier_rmse_l):0.3f}')

    return tuples


def generate_image_meta_pickle(dataset_root: str):
    cam_params_path = dataset_root + '/cam_params/'
    factor_x = 224. / 600.
    factor_y = 384. / 900.

    K1 = np.loadtxt(cam_params_path + 'K_cam1.csv', delimiter=',')
    fx = K1[0][0]
    fy = K1[1][1]
    cx = K1[0][2]
    cy = K1[1][2]

    cy = 1232. - cy 
    cx -= 400.  # cx
    cy -= 182.  # cy
    cx = cx * factor_x
    cy = cy * factor_y

    K1[0][0] = fy * factor_y
    K1[0][2] = cy
    K1[1][1] = fx * factor_x
    K1[1][2] = cx

    K2 = np.loadtxt(cam_params_path + 'K_cam2.csv', delimiter=',')
    fx = K2[0][0]
    fy = K2[1][1]
    cx = K2[0][2]
    cy = K2[1][2]

    cy = 1232. - cy 
    cx -= 400.  # cx
    cy -= 182.  # cy
    cx = cx * factor_x
    cy = cy * factor_y

    K2[0][0] = fy * factor_y
    K2[0][2] = cy
    K2[1][1] = fx * factor_x
    K2[1][2] = cx

    K3 = np.loadtxt(cam_params_path + 'K_cam3.csv', delimiter=',')
    fx = K3[0][0]
    fy = K3[1][1]
    cx = K3[0][2]
    cy = K3[1][2]

    cy = 1232. - cy 
    cx -= 400.  # cx
    cy -= 182.  # cy
    cx = cx * factor_x
    cy = cy * factor_y

    K3[0][0] = fy * factor_y
    K3[0][2] = cy
    K3[1][1] = fx * factor_x
    K3[1][2] = cx

    K4 = np.loadtxt(cam_params_path + 'K_cam4.csv', delimiter=',')
    fx = K4[0][0]
    fy = K4[1][1]
    cx = K4[0][2]
    cy = K4[1][2]

    cy = 1232. - cy 
    cx -= 400.  # cx
    cy -= 182.  # cy
    cx = cx * factor_x
    cy = cy * factor_y

    K4[0][0] = fy * factor_y
    K4[0][2] = cy
    K4[1][1] = fx * factor_x
    K4[1][2] = cx

    K5 = np.loadtxt(cam_params_path + 'K_cam5.csv', delimiter=',')
    fx = K5[0][0]
    fy = K5[1][1]
    cx = K5[0][2]
    cy = K5[1][2]
    
    cy = 1232. - cy 
    cx -= 400.  # cx
    cy -= 182.  # cy
    cx = cx * factor_x
    cy = cy * factor_y

    K5[0][0] = fy * factor_y
    K5[0][2] = cy
    K5[1][1] = fx * factor_x
    K5[1][2] = cx
    x_lb3_c0 = np.loadtxt(cam_params_path + 'x_lb3_c0.csv', delimiter=',')
    x_lb3_c1 = np.loadtxt(cam_params_path + 'x_lb3_c1.csv', delimiter=',')
    x_lb3_c2 = np.loadtxt(cam_params_path + 'x_lb3_c2.csv', delimiter=',')
    x_lb3_c3 = np.loadtxt(cam_params_path + 'x_lb3_c3.csv', delimiter=',')
    x_lb3_c4 = np.loadtxt(cam_params_path + 'x_lb3_c4.csv', delimiter=',')
    x_lb3_c5 = np.loadtxt(cam_params_path + 'x_lb3_c5.csv', delimiter=',')

    x_body_lb3 = [0.035, 0.002, -1.23, -179.93, -0.23, 0.50]
    T0 = calculate_T(x_lb3_c0, x_body_lb3)
    T1 = calculate_T(x_lb3_c1, x_body_lb3)
    T2 = calculate_T(x_lb3_c2, x_body_lb3)
    T3 = calculate_T(x_lb3_c3, x_body_lb3)
    T4 = calculate_T(x_lb3_c4, x_body_lb3)
    T5 = calculate_T(x_lb3_c5, x_body_lb3)

    print(T1.shape)
    image_meta = {"K": [K1, K2, K3, K4, K5], "T": [T1, T2, T3, T4, T5]}
    with open(cam_params_path + 'image_meta.pkl', 'wb') as handle:
        pickle.dump(image_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training tuples')
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--pos_threshold', default=2.0)
    parser.add_argument('--neg_threshold', default=3.0)
    parser.add_argument('--sampling_distance', type=float, default=0.2)
    args = parser.parse_args()

    sequences = ['2012-02-04', '2012-03-17']

    print(f'Dataset root: {args.dataset_root}')
    print(f'Sequences: {sequences}')
    print(f'Threshold for positive examples: {args.pos_threshold}')
    print(f'Threshold for negative examples: {args.neg_threshold}')
    print(f'Minimum displacement between consecutive anchors: {args.sampling_distance}')

    ds = NCLTSequences(args.dataset_root, sequences, split='train', sampling_distance=args.sampling_distance)
    train_tuples = generate_training_tuples(ds, args.pos_threshold, args.neg_threshold)
    pickle_name = f'train_{sequences[0]}_{sequences[1]}_{args.pos_threshold}_{args.neg_threshold}.pickle'
    train_tuples_filepath = os.path.join(args.dataset_root, pickle_name)
    # pickle.dump(train_tuples, open(train_tuples_filepath, 'wb'))
    train_tuples = None

    ds = NCLTSequences(args.dataset_root, sequences, split='test', sampling_distance=args.sampling_distance)
    print("test sequences len: ", len(ds))
    test_tuples = generate_training_tuples(ds, args.pos_threshold, args.neg_threshold)
    print("test tuple length: ", len(test_tuples))
    pickle_name = f'val_{sequences[0]}_{sequences[1]}_{args.pos_threshold}_{args.neg_threshold}.pickle'
    test_tuples_filepath = os.path.join(args.dataset_root, pickle_name)
    # pickle.dump(test_tuples, open(test_tuples_filepath, 'wb'))

    # generate_image_meta_pickle(args.dataset_root)