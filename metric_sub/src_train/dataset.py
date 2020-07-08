import os
import random
import json
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2

from read_video import faster_read_frame_at_index
from augmentation import *

standardize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class TaobaoTrainSet(Dataset):
    # (width, height)
    size_template = {'S': [(80, 320), (106, 320), (134, 320), (160, 320), (170, 298), (186, 280), (214, 266), (234, 234), 
                           (266, 214), (280, 186), (298, 170), (320, 160)],
                     'SM': [(100, 400), (134, 400), (166, 400), (200, 400), (214, 374), (234, 350), (256, 320), (290, 290),
                            (320, 256), (350, 234), (374, 214), (400, 200)],
                     'M': [(120, 480), (160, 480), (200, 480), (240, 480), (256, 448), (280, 420), (320, 400), (352, 352), 
                           (400, 320), (420, 280), (448, 256), (480, 240)],
                     'C': [(234, 234), (234, 234), (234, 234), (234, 234), (234, 234), (234, 234), (234, 234), (234, 234),
                           (234, 234), (234, 234), (234, 234), (234, 234)]
                    }

    def __init__(self, json_list, scale='S', mode='train'):
        self.json_list = json_list
        self.scale = scale
        self.sizes = self.size_template[scale]
        self.augmentor = aug_medium()
        self.mode = mode

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, idx):
        dataset_dict = self.json_list[idx] # dict of single instance of an image/video

        if dataset_dict['type'] == 'image':
            image = cv2.imread(dataset_dict["file_name"])
        elif dataset_dict['type'] == 'video':
            image = faster_read_frame_at_index(dataset_dict["file_name"], dataset_dict["frame"])


        image = get_cropped_img_fast(image, dataset_dict['bbox'])
        new_size = self.sizes[dataset_dict['aspect_group']]
        image = cv2.resize(image, new_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode == 'train':
            data_dict = {"image": image}
            augmented = self.augmentor(**data_dict)
            image = augmented["image"]

            if random.random() < 0.7:
                image = random_affine(image, degrees=8, translate=.0625, scale=.1, shear=8)

            if random.random() < 0.7:
                image = cutout(image, int(new_size[1]*0.63), int(new_size[0]*0.63), fill_value=114)

        # if random.randint(0, 1) == 0:
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = standardize(transforms.functional.to_tensor(image))
        return image, dataset_dict['instance_id'], dataset_dict['category_id']