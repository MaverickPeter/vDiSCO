import numpy as np
import os
from scipy.spatial import distance_matrix
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy.matlib as matlib
from math import sin, cos, atan2, sqrt


# Faulty point clouds (with 0 points)
FAULTY_POINTCLOUDS = []

# Coordinates of test region centres (in Oxford sequences)
TEST_REGION_CENTRES = np.array([[5735400, 620000]])

# Radius of the test region
TEST_REGION_RADIUS = 220

# Boundary between training and test region - to ensure there's no overlap between training and test clouds
TEST_TRAIN_BOUNDARY = 50


def in_train_split(pos):
    # returns true if pos is in train split
    assert pos.ndim == 2
    assert pos.shape[1] == 2
    dist = distance_matrix(pos, TEST_REGION_CENTRES)
    mask = (dist > TEST_REGION_RADIUS + TEST_TRAIN_BOUNDARY).all(axis=1)
    return mask


def in_test_split(pos):
    # returns true if position is in evaluation split
    assert pos.ndim == 2
    assert pos.shape[1] == 2
    dist = distance_matrix(pos, TEST_REGION_CENTRES)
    mask = (dist < TEST_REGION_RADIUS).any(axis=1)
    return mask


def find_nearest_ndx(ts, timestamps):
    ndx = np.searchsorted(timestamps, ts)
    if ndx == 0:
        return ndx
    elif ndx == len(timestamps):
        return ndx - 1
    else:
        assert timestamps[ndx-1] <= ts <= timestamps[ndx]
        if ts - timestamps[ndx-1] < timestamps[ndx] - ts:
            return ndx - 1
        else:
            return ndx


def read_ts_file(ts_filepath: str):
    with open(ts_filepath, "r") as h:
        txt_ts = h.readlines()

    n = len(txt_ts)
    ts = np.zeros((n,), dtype=np.int64)

    for ndx, timestamp in enumerate(txt_ts):
        # Split by comma and remove whitespaces
        temp = [e.strip() for e in timestamp.split(' ')]
        assert len(temp) == 2, f'Invalid line in timestamp file: {temp}'

        ts[ndx] = int(temp[0])

    return ts


def read_lidar_poses(poses_filepath: str, left_lidar_filepath: str, pose_time_tolerance: float = 1.):
    # Read global poses from .csv file and link each lidar_scan with the nearest pose
    # threshold: threshold in seconds
    # Returns a dictionary with (4, 4) pose matrix indexed by a timestamp (as integer)    

    with open(poses_filepath, "r") as h:
        txt_poses = h.readlines()

    n = len(txt_poses)
    system_timestamps = np.zeros((n,), dtype=np.int64)
    poses = np.zeros((n, 4, 4), dtype=np.float64)       # 4x4 pose matrix

    for ndx, pose in enumerate(txt_poses):
        # Split by comma and remove whitespaces
        temp = [e.strip() for e in pose.split(',')]
        if ndx == 0:
            continue
        ndx -= 1
        assert len(temp) == 15, f'Invalid line in global poses file: {temp}'
        system_timestamps[ndx] = int(temp[0])
        poses[ndx] = RPY2Rot(float(temp[5]), float(temp[6]), float(temp[7]), float(temp[12]), float(temp[13]), float(temp[14]))

    # Ensure timestamps and poses are sorted in ascending order
    sorted_ndx = np.argsort(system_timestamps, axis=0)
    system_timestamps = system_timestamps[sorted_ndx]
    poses = poses[sorted_ndx]

    # List LiDAR scan timestamps
    left_lidar_timestamps = [int(os.path.splitext(f)[0]) for f in os.listdir(left_lidar_filepath) if
                            os.path.splitext(f)[1] == '.bin']
    left_lidar_timestamps.sort()

    lidar_timestamps = []
    lidar_poses = []
    count_rejected = 0

    for ndx, lidar_ts in enumerate(left_lidar_timestamps):
        # Skip faulty point clouds
        if lidar_ts in FAULTY_POINTCLOUDS:
            continue

        # Find index of the closest timestamp
        closest_ts_ndx = find_nearest_ndx(lidar_ts, system_timestamps)
        delta = abs(system_timestamps[closest_ts_ndx] - lidar_ts)
        # Timestamp is in nanoseconds = 1e-9 second
        if delta > pose_time_tolerance * 1000000000:
            # Reject point cloud without corresponding pose
            count_rejected += 1
            continue

        lidar_timestamps.append(lidar_ts)
        lidar_poses.append(poses[closest_ts_ndx])

    lidar_timestamps = np.array(lidar_timestamps, dtype=np.int64)
    lidar_poses = np.array(lidar_poses, dtype=np.float64)     # (northing, easting) position

    print(f'{len(lidar_timestamps)} scans with valid pose, {count_rejected} rejected due to unknown pose')
    return lidar_timestamps, lidar_poses


def relative_pose(m1, m2):
    # SE(3) pose is 4x 4 matrix, such that
    # Pw = [R | T] @ [P]
    #      [0 | 1]   [1]
    # where Pw are coordinates in the world reference frame and P are coordinates in the camera frame
    # m1: coords in camera/lidar1 reference frame -> world coordinate frame
    # m2: coords in camera/lidar2 coords -> world coordinate frame
    # returns: relative pose of the first camera with respect to the second camera
    #          transformation matrix to convert coords in camera/lidar1 reference frame to coords in
    #          camera/lidar2 reference frame
    #
    m = np.linalg.inv(m2) @ m1
    # !!!!!!!!!! Fix for relative pose !!!!!!!!!!!!!
    m[:3, 3] = -m[:3, 3]
    return m


def RPY2Rot(x, y, z, roll, pitch, yaw):
    R = [[np.cos(pitch)*np.cos(yaw), -np.cos(pitch)*np.sin(yaw), np.sin(pitch), x],
         [np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(roll)*np.sin(yaw),
          -np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(roll)*np.cos(yaw),
          -np.sin(roll)*np.cos(pitch), y],
         [-np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll),
          np.cos(roll)*np.sin(pitch)*np.sin(yaw) + np.sin(roll)*np.cos(yaw),
          np.cos(roll)*np.cos(pitch), z],
         [0, 0, 0, 1]]
    return np.array(R) 

# apply random rotation along z axis
def random_rotation(xyz, angle_range=(-np.pi, np.pi)):
    angle = np.random.uniform(*angle_range)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]]).transpose()
    return np.dot(xyz, rotation_matrix)


# get the 4x4 SE(3) transformation matrix from euler angles
def euler2se3(x, y, z, roll, pitch, yaw):
    se3 = np.eye(4, dtype=np.float64)   
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    se3[:3, :3] = R
    se3[:3, 3] = np.array([x, y, z])

    return se3

# apply 4x4 SE(3) transformation matrix on (N, 3) point cloud or 3x3 transformation on (N, 2) point cloud
def apply_transform(pc: torch.Tensor, m: torch.Tensor):
    assert pc.ndim == 2
    n_dim = pc.shape[1]
    assert n_dim == 2 or n_dim == 3
    assert m.shape == (n_dim + 1, n_dim + 1)
    # (m @ pc.t).t = pc @ m.t
    pc = pc @ m[:n_dim, :n_dim].transpose(1, 0) + m[:n_dim, -1]

    return pc

################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

import numpy as np
import numpy.matlib as matlib
from math import sin, cos, atan2, sqrt

MATRIX_MATCH_TOLERANCE = 1e-4


def build_se3_transform(xyzrpy):
    """Creates an SE3 transform from translation and Euler angles.

    Args:
        xyzrpy (list[float]): translation and Euler angles for transform. Must have six components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: SE3 homogeneous transformation matrix

    Raises:
        ValueError: if `len(xyzrpy) != 6`

    """
    if len(xyzrpy) != 6:
        raise ValueError("Must supply 6 values to build transform")

    se3 = matlib.identity(4)
    se3[0:3, 0:3] = euler_to_so3(xyzrpy[3:6])
    se3[0:3, 3] = np.matrix(xyzrpy[0:3]).transpose()
    return se3


def euler_to_so3(rpy):
    """Converts Euler angles to an SO3 rotation matrix.

    Args:
        rpy (list[float]): Euler angles (in radians). Must have three components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: 3x3 SO3 rotation matrix

    Raises:
        ValueError: if `len(rpy) != 3`.

    """
    if len(rpy) != 3:
        raise ValueError("Euler angles must have three components")

    R_x = np.matrix([[1, 0, 0],
                     [0, cos(rpy[0]), -sin(rpy[0])],
                     [0, sin(rpy[0]), cos(rpy[0])]])
    R_y = np.matrix([[cos(rpy[1]), 0, sin(rpy[1])],
                     [0, 1, 0],
                     [-sin(rpy[1]), 0, cos(rpy[1])]])
    R_z = np.matrix([[cos(rpy[2]), -sin(rpy[2]), 0],
                     [sin(rpy[2]), cos(rpy[2]), 0],
                     [0, 0, 1]])
    R_zyx = R_z * R_y * R_x
    return R_zyx


def so3_to_euler(so3):
    """Converts an SO3 rotation matrix to Euler angles

    Args:
        so3: 3x3 rotation matrix

    Returns:
        numpy.matrixlib.defmatrix.matrix: list of Euler angles (size 3)

    Raises:
        ValueError: if so3 is not 3x3
        ValueError: if a valid Euler parametrisation cannot be found

    """
    if so3.shape != (3, 3):
        raise ValueError("SO3 matrix must be 3x3")
    roll = atan2(so3[2, 1], so3[2, 2])
    yaw = atan2(so3[1, 0], so3[0, 0])
    denom = sqrt(so3[0, 0] ** 2 + so3[1, 0] ** 2)
    pitch_poss = [atan2(-so3[2, 0], denom), atan2(-so3[2, 0], -denom)]

    R = euler_to_so3((roll, pitch_poss[0], yaw))

    if (so3 - R).sum() < MATRIX_MATCH_TOLERANCE:
        return np.matrix([roll, pitch_poss[0], yaw])
    else:
        R = euler_to_so3((roll, pitch_poss[1], yaw))
        if (so3 - R).sum() > MATRIX_MATCH_TOLERANCE:
            raise ValueError("Could not find valid pitch angle")
        return np.matrix([roll, pitch_poss[1], yaw])


def so3_to_quaternion(so3):
    """Converts an SO3 rotation matrix to a quaternion

    Args:
        so3: 3x3 rotation matrix

    Returns:
        numpy.ndarray: quaternion [w, x, y, z]

    Raises:
        ValueError: if so3 is not 3x3
    """
    if so3.shape != (3, 3):
        raise ValueError("SO3 matrix must be 3x3")

    R_xx = so3[0, 0]
    R_xy = so3[0, 1]
    R_xz = so3[0, 2]
    R_yx = so3[1, 0]
    R_yy = so3[1, 1]
    R_yz = so3[1, 2]
    R_zx = so3[2, 0]
    R_zy = so3[2, 1]
    R_zz = so3[2, 2]

    try:
        w = sqrt(so3.trace() + 1) / 2
    except(ValueError):
        # w is non-real
        w = 0

    # Due to numerical precision the value passed to `sqrt` may be a negative of the order 1e-15.
    # To avoid this error we clip these values to a minimum value of 0.
    x = sqrt(max(1 + R_xx - R_yy - R_zz, 0)) / 2
    y = sqrt(max(1 + R_yy - R_xx - R_zz, 0)) / 2
    z = sqrt(max(1 + R_zz - R_yy - R_xx, 0)) / 2

    max_index = max(range(4), key=[w, x, y, z].__getitem__)

    if max_index == 0:
        x = (R_zy - R_yz) / (4 * w)
        y = (R_xz - R_zx) / (4 * w)
        z = (R_yx - R_xy) / (4 * w)
    elif max_index == 1:
        w = (R_zy - R_yz) / (4 * x)
        y = (R_xy + R_yx) / (4 * x)
        z = (R_zx + R_xz) / (4 * x)
    elif max_index == 2:
        w = (R_xz - R_zx) / (4 * y)
        x = (R_xy + R_yx) / (4 * y)
        z = (R_yz + R_zy) / (4 * y)
    elif max_index == 3:
        w = (R_yx - R_xy) / (4 * z)
        x = (R_zx + R_xz) / (4 * z)
        y = (R_yz + R_zy) / (4 * z)

    return np.array([w, x, y, z])


def se3_to_components(se3):
    """Converts an SE3 rotation matrix to linear translation and Euler angles

    Args:
        se3: 4x4 transformation matrix

    Returns:
        numpy.matrixlib.defmatrix.matrix: list of [x, y, z, roll, pitch, yaw]

    Raises:
        ValueError: if se3 is not 4x4
        ValueError: if a valid Euler parametrisation cannot be found

    """
    if se3.shape != (4, 4):
        raise ValueError("SE3 transform must be a 4x4 matrix")
    xyzrpy = np.empty(6)
    xyzrpy[0:3] = se3[0:3, 3].transpose()
    xyzrpy[3:6] = so3_to_euler(se3[0:3, 0:3])
    return xyzrpy


