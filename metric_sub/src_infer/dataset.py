import cv2
import copy
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import cv2
from read_video import faster_read_frame_at_index

standardize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_cropped_img_fast(image, bbox):
    x0, y0, x1, y1 = bbox
    return image[y0:y1, x0:x1]

def inst_process(img, bbox, aspect_group, scale='S'):
    size_template = {'S': [(80, 320), (106, 320), (134, 320), (160, 320), (170, 298), (186, 280), (214, 266), (234, 234), 
                           (266, 214), (280, 186), (298, 170), (320, 160)],
                     'SM': [(100, 400), (134, 400), (166, 400), (200, 400), (214, 374), (234, 350), (256, 320), (290, 290),
                            (320, 256), (350, 234), (374, 214), (400, 200)],
                     'M': [(120, 480), (160, 480), (200, 480), (240, 480), (256, 448), (280, 420), (320, 400), (352, 352), 
                           (400, 320), (420, 280), (448, 256), (480, 240)],
                     'C': [(234, 234), (234, 234), (234, 234), (234, 234), (234, 234), (234, 234), (234, 234), (234, 234),
                           (234, 234), (234, 234), (234, 234), (234, 234)]
                    }
    sizes = size_template[scale]

    inst = get_cropped_img_fast(img, bbox)
    inst = cv2.resize(inst, sizes[aspect_group])
    inst = cv2.cvtColor(inst, cv2.COLOR_BGR2RGB)
    inst = standardize(transforms.functional.to_tensor(inst))
    return inst


class MetricInferSet(Dataset):
    
    def __init__(self, instance_list, scale='SM'):
        self.instance_list = instance_list
        self.scale = scale
        
    def __len__(self):
        return len(self.instance_list)
    
    def __getitem__(self, idx):
        inst_dict = self.instance_list[idx]

        if inst_dict['type'] == 'image':
            img = cv2.imread(inst_dict['file_name'])
        elif inst_dict['type'] == 'video':
            img = faster_read_frame_at_index(inst_dict["file_name"], inst_dict["frame"])

        inst = inst_process(img, inst_dict['bbox'], inst_dict['aspect_group'], scale=self.scale)
        
        return inst.to(DEVICE), idx