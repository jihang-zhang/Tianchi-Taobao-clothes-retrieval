import os
from os.path import sep, join
import json
from tqdm import tqdm

import cv2
import imagesize


def pjoin(*args, **kwargs):
    return join(*args, **kwargs).replace(sep, '/')

def get_img_dicts(input_dir):

    result = []
    for item_id in tqdm(os.listdir(input_dir)):
        for img in os.listdir(pjoin(input_dir, item_id)):
            image_dict = {}

            image_dict['file_name'] = pjoin(input_dir, item_id, img)
            w, h = imagesize.get(image_dict['file_name'])
            image_dict['height'] = h
            image_dict['width'] = w
            image_dict['item_id'] = item_id
            image_dict['type'] = 'image'

            result.append(image_dict)

    return result

def get_vid_dicts(input_dir):

    result = []
    for vid_name in tqdm(os.listdir(input_dir)):
        item_id = vid_name[:-4]
        image_dict = {}

        image_dict['file_name'] = pjoin(input_dir, vid_name)
        capture = cv2.VideoCapture(image_dict['file_name'])
        w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        image_dict['height'] = h
        image_dict['width'] = w
        image_dict['item_id'] = item_id
        image_dict['type'] = 'video'

        result.append(image_dict)

    return result

# d = get_img_dicts('../input/validation_dataset_part1')