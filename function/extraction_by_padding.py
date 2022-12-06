import cv2 as ccc
import glob
from Train_Mask_RCNN_Image_extraction import *
from mrcnn.visualize import random_colors,get_mask_contours,draw_mask
import os
rootpath=os.path.abspath("D:\\zeeshan new data thesisi\\pycharm\\extraction")
def part_extraction(img,weight_path):
    test_model, inference_config = load_inference_model(1, weight_path)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r = test_model.detect([img], verbose=1)[0]
    box=r['rois']
    y1,x1,y2,x2=box[0]
    im=img[y1:y2,x1:x2]
    return im
def add_padding(img,size):
    old_image_height, old_image_width, channels = img.shape
    new_image_width = size
    new_image_height = size
    color = (0, 0, 0)
    result = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2
    result[y_center:y_center + old_image_height, x_center:x_center + old_image_width] = img
    return result