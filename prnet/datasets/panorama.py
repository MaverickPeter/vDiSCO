import pickle
import numpy as np
import argparse
from torchvision import datasets
from torchvision import transforms
import imutils
import cv2
import os
import torch

def generate_sph_image(images, dataset_type, prespheredir):
    stitcher = Stitcher(dataset_type)
    sph_img = stitcher.stitch(images, prespheredir)
    sph_img = cv2.cvtColor(sph_img, cv2.COLOR_BGR2RGB)
    sph_img = cv2.rotate(sph_img, cv2.ROTATE_180)
    return sph_img

def remove_the_blackborder(image):

    img = cv2.medianBlur(image, 5) 
    b = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)
    binary_image = b[1]
    binary_image = cv2.cvtColor(binary_image,cv2.COLOR_BGR2GRAY)
 
    edges_y, edges_x = np.where(binary_image==255) ##h, w
    bottom = min(edges_y)             
    top = max(edges_y) 
    height = top - bottom            
                                   
    left = min(edges_x)           
    right = max(edges_x)             
    height = top - bottom 
    width = right - left

    res_image = image[bottom:bottom+height, left:left+width]

    return res_image   


def XYZtoRC(dataset_type, hits, img, cam_id):
    if dataset_type == 'nclt':
        with open("/mnt/workspace/datasets/NCLT/cam_params/image_meta.pkl", 'rb') as handle:
            image_meta = pickle.load(handle)
    elif dataset_type == 'oxford':
        with open("/mnt/workspace/datasets/Oxford/image_meta.pkl", 'rb') as handle:
            image_meta = pickle.load(handle)

    intrins = np.array(image_meta['K'])
    cams_T_body = np.array(image_meta['T'])
    # print(hits)
    K = intrins[cam_id]
    T = cams_T_body[cam_id]

    hits_c = np.matmul(T, hits)
    hits_im = np.matmul(K, hits_c[0:3, :])

    x_im = hits_im[0, :] / (hits_im[2, :] + 1e-8)
    y_im = hits_im[1, :] / (hits_im[2, :] + 1e-8)
    z_im = hits_im[2, :]
    idx_infront = (z_im > 0) & (x_im > 0) & (x_im < img.shape[1]) & (y_im > 0) & (y_im < img.shape[0])
    y_im = y_im[idx_infront]
    x_im = x_im[idx_infront]

    idx = (y_im.astype(int), x_im.astype(int))

    return idx, idx_infront


def spherical_projection(dataset_type, imgs, f, prespheredir) :
    R = f
    rows = imgs[0].shape[0]
    cols = imgs[0].shape[1]

    outrow = 2*rows
    outcol = 2*cols

    panoramic = np.zeros((outrow, outcol, 3))

    points = []
    pix_x = []
    pix_y = []
    
    # for y in range(outrow):
    #    for x in range(outcol):
    #         theta = -(2 * np.pi * x / (outcol-1) - np.pi)
    #         phi = np.pi * y / (outrow-1)

    #         globalZ = R * np.cos(phi) 
    #         globalX = R * np.sin(phi) * np.cos(theta)
    #         globalY = R * np.sin(phi) * np.sin(theta)
    #         points += [[globalX, globalY, globalZ, 1]]
    #         pix_x.append(x)
    #         pix_y.append(y)

    # points = np.asarray(points)
    # points = points.transpose()
    # pix_x = np.asarray(pix_x)
    # pix_y = np.asarray(pix_y)
    # pix = np.asarray([pix_y, pix_x])

    # np.save("/mnt/workspace/datasets/NCLT/sphere_points_large.npy", points)
    # np.save("/mnt/workspace/datasets/NCLT/pix_large.npy", pix)

    points = np.load(os.path.join(prespheredir, "sphere_points.npy"))
    pix = np.load(os.path.join(prespheredir, "pix.npy"))
    pix_x = pix[1]
    pix_y = pix[0]

    # print("====> sphere points sampling done.")

    for cam_id in range(len(imgs)):
        idx, valid = XYZtoRC(dataset_type, points, imgs[cam_id], cam_id)

        rgb = imgs[cam_id][idx]
        valid_y = pix_y[valid]
        valid_x = pix_x[valid]

        if dataset_type == 'nclt':
            nonblack = np.where(rgb > 10, 1, 0)
            nonblack = np.sum(nonblack, axis=1)
            nonblack = np.where(nonblack > 2, True, False)

            panoramic[valid_y[nonblack], valid_x[nonblack], 0] = rgb[nonblack,0]
            panoramic[valid_y[nonblack], valid_x[nonblack], 1] = rgb[nonblack,1]
            panoramic[valid_y[nonblack], valid_x[nonblack], 2] = rgb[nonblack,2]
        elif dataset_type == 'oxford':
            panoramic[valid_y, valid_x, 0] = rgb[...,0]
            panoramic[valid_y, valid_x, 1] = rgb[...,1]
            panoramic[valid_y, valid_x, 2] = rgb[...,2]

        cv_img = cv2.merge([panoramic[...,0], panoramic[...,1], panoramic[...,2]])
        cv_img = cv_img.astype(np.uint8)

    return panoramic, cv_img


class Stitcher:
    def __init__(self, dataset_type):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3(or_better=True)
        self.dataset_type = dataset_type

    def stitch(self, images, prespheredir):

        if self.dataset_type == 'nclt':
            with open("/mnt/workspace/datasets/NCLT/cam_params/image_meta.pkl", 'rb') as handle:
                image_meta = pickle.load(handle)
        elif self.dataset_type == 'oxford':
            with open("/mnt/workspace/datasets/Oxford/image_meta.pkl", 'rb') as handle:
                image_meta = pickle.load(handle)

        intrins = np.array(image_meta['K'])
        cams_T_body = np.array(image_meta['T'])
        body_T_cam3 = np.linalg.inv(cams_T_body[3])

        focal = intrins[3][1,1]

        sph_img, cv_img = spherical_projection(self.dataset_type, images, focal, prespheredir)
        # sph_img = remove_the_blackborder(cv_img)

        if self.dataset_type == 'nclt':
            sph_img = cv_img[170:330,...]
            # cv2.imwrite("/mnt/workspace/sph.png",sph_img)
            # sph_img = sph_img
        elif self.dataset_type == 'oxford':
            sph_img = sph_img[200:400,...]

        return sph_img


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True,
                    help="path to the images dir")

    args = vars(ap.parse_args())

    imagePaths = sorted(list(paths.list_images(args["dir"])))

    images = []

    # load the images
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        images.append(image)

    # stitch the images together to create a panorama
    stitcher = Stitcher()
    result = stitcher.stitch(images, "/mnt/workspace/datasets/NCLT/")
    result = cv2.rotate(result, cv2.ROTATE_180)
    cv2.imwrite("/mnt/workspace/code/panorama_pyimagesearch/output.jpg", result)
