import cv2
import os
from tqdm import tqdm

for i in range(5):
    print(i)
    i = i+1
    map_file = "/mnt/workspace/datasets/NCLT/U2D/U2D_Cam" + str(i) + "_1616X1232.txt"
    camera_path = "/mnt/workspace/datasets/NCLT/2012-02-04/lb3_u/Cam" + str(i) + "/"
    camera_save_path = "/mnt/workspace/datasets/NCLT/2012-02-04/lb3_u_s_384/Cam" + str(i) + "/"
    if not os.path.exists(camera_save_path):
        os.makedirs(camera_save_path)
    image_filenames = sorted(os.listdir(camera_path))
    for image in tqdm(image_filenames):
        im = cv2.imread(camera_path + image)
        input_image = im[150:150+900, 400:400+600, :]
        input_image = cv2.resize(input_image, (224, 384))
        image_name = image.split(".")[0]
        filename = camera_save_path + image_name + ".jpg"
        cv2.imwrite(filename, input_image)
