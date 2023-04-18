# !/usr/bin/python
#
# Demonstrates how to project velodyne points to camera imagery. Requires a binary
# velodyne sync file, undistorted image, and assumes that the calibration files are
# in the directory.
#
# To use:
#
#    python project_vel_to_cam.py vel img cam_num
#
#       vel:  The velodyne binary file (timestamp.bin)
#       img:  The undistorted image (timestamp.tiff)
#   cam_num:  The index (0 through 5) of the camera
#

import sys
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# from mmcv.ops import Voxelization

#from undistort import *

def convert(x_s, y_s, z_s):

    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z

def load_vel_hits(filename):

    f_bin = open(filename, "rb")

    hits = []

    while True:

        x_str = f_bin.read(2)
        if x_str == b'': # eof
            break

        x = struct.unpack('<H', x_str)[0]
        y = struct.unpack('<H', f_bin.read(2))[0]
        z = struct.unpack('<H', f_bin.read(2))[0]
        i = struct.unpack('B', f_bin.read(1))[0]
        l = struct.unpack('B', f_bin.read(1))[0]

        x, y, z = convert(x, y, z)
        s = "%5.3f, %5.3f, %5.3f, %d, %d" % (x, y, z, i, l)

        # Load in homogenous
        hits += [[x, y, z, 1]]

    f_bin.close()
    hits = np.asarray(hits)
    # hits[:,2] = -hits[:,2]

    print("height median:", np.median(hits[:,2]))

    return hits.transpose()

def ssc_to_homo(ssc):

    # Convert 6-DOF ssc coordinate transformation to 4x4 homogeneous matrix
    # transformation

    sr = np.sin(np.pi/180.0 * ssc[3])
    cr = np.cos(np.pi/180.0 * ssc[3])

    sp = np.sin(np.pi/180.0 * ssc[4])
    cp = np.cos(np.pi/180.0 * ssc[4])

    sh = np.sin(np.pi/180.0 * ssc[5])
    ch = np.cos(np.pi/180.0 * ssc[5])

    H = np.zeros((4, 4))

    H[0, 0] = ch*cp
    H[0, 1] = -sh*cr + ch*sp*sr
    H[0, 2] = sh*sr + ch*sp*cr
    H[1, 0] = sh*cp
    H[1, 1] = ch*cr + sh*sp*sr
    H[1, 2] = -ch*sr + sh*sp*cr
    H[2, 0] = -sp
    H[2, 1] = cp*sr
    H[2, 2] = cp*cr

    H[0, 3] = ssc[0]
    H[1, 3] = ssc[1]
    H[2, 3] = ssc[2]

    H[3, 3] = 1

    return H

def project_vel_to_cam(hits, cam_num):

    # Load camera parameters
    K = np.loadtxt('/mnt/workspace/datasets/NCLT/cam_params/K_cam%d.csv' % (cam_num), delimiter=',')
    factor_x = 224. / 600.
    factor_y = 384. / 900.
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    cx = 1616. - cx
    cx -= 616.  # cx
    cy -= 150.  # cy
    cx = cx * factor_x
    cy = cy * factor_y

    K[0][0] = fy * factor_y
    K[0][2] = cy
    K[1][1] = fx * factor_x
    K[1][2] = cx
    x_lb3_c = np.loadtxt('/mnt/workspace/datasets/NCLT/cam_params/x_lb3_c%d.csv' % (cam_num), delimiter=',')

    # Other coordinate transforms we need
    x_body_lb3 = [0.035, 0.002, -1.23, -179.93, -0.23, 0.50]

    x_camNormal_cam = [0.0, 0.0, 0.0, 0.0, 0.0, -90]
    T_camNormal_cam = ssc_to_homo(x_camNormal_cam)

    # Now do the projection
    T_lb3_c = ssc_to_homo(x_lb3_c)
    T_body_lb3 = ssc_to_homo(x_body_lb3)

    T_lb3_body = np.linalg.inv(T_body_lb3)
    T_c_lb3 = np.linalg.inv(T_lb3_c)

    T_c_body = np.matmul(T_c_lb3, T_lb3_body)
    T_body_c = np.linalg.inv(T_c_body)

    T_camNormal_body = np.matmul(T_camNormal_cam, T_c_body)
    print(T_camNormal_body)
    hits_c = np.matmul(T_camNormal_body, hits)
    hits_im = np.matmul(K, hits_c[0:3, :])

    return hits_im

def load_im_file_for_generate(filename):
    input_image = cv2.imread(filename)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # input_image = input_image[150:150 + 900, 400:400 + 600, :]
    # input_image = cv2.resize(input_image, (240, 360))
    return input_image

def main(args):

    if len(args)<4:
        print("""Incorrect usage.

To use:

   python project_vel_to_cam.py vel img cam_num

      vel:  The velodyne binary file (timestamp.bin)
      img:  The undistorted image (timestamp.tiff)
  cam_num:  The index (0 through 5) of the camera
""")
        return 1

    # image = mpimg.imread(args[2])
    image = load_im_file_for_generate(args[2])
    cam_num = int(args[3])
    # Load velodyne points
    hits_body = load_vel_hits(args[1])
    print(hits_body.shape)

    hits_image = project_vel_to_cam(hits_body, cam_num)
    print(hits_image.shape)

    x_im = hits_image[0, :]/hits_image[2, :]
    y_im = hits_image[1, :]/hits_image[2, :]
    z_im = hits_image[2, :]

    idx_infront = (z_im > 0) & (x_im > 0) & (x_im < 384) & (y_im > 0) & (y_im < 224)
    # idx_infront = (z_im > 0) & (x_im > 0) & (x_im < 1232) & (y_im > 0) & (y_im < 1616)

    hits_body_image2 = hits_body.transpose()[idx_infront]

    x_im = x_im[idx_infront]
    y_im = y_im[idx_infront]
    z_im = z_im[idx_infront]

    x_im_int = x_im.astype(np.int32)
    y_im_int = y_im.astype(np.int32)
    z_im_int = z_im.astype(np.int32)
    print(x_im_int.shape)
    image = np.asarray(image)

    plt.figure(1)
    plt.imshow(image)
    plt.scatter(x_im, y_im, c=z_im%20.0/20.0, cmap='jet', alpha=0.1, s=1)
    # plt.xlim(0, 240)
    # plt.ylim(0, 360)
    plt.show()
    plt.savefig("/mnt/workspace/proj_rect.jpg")

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
