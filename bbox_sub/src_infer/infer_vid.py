import json
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from detectron2.config import get_cfg
from detectron2.modeling import build_model, GeneralizedRCNNWithTTA
from detectron2.checkpoint import DetectionCheckpointer

from config import add_vovnet_config
from vovnet import *
from network import ENet
from prepare_json import get_vid_dicts
from read_video import faster_read_frame_at_index
from make_dataloader import make_dataloader
from dataset import inst_process

DEBUG = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def infer_vid(bbox_net, dataset, frames=list(range(80, 400, 20))):

    # infer bbox
    aspect_template = np.array([0.25, 0.335, 0.415, 0.5, 0.5721925, 0.66857143, 0.8, 1., 1.25, 1.4957264, 1.74766355, 2.])

    bbox_net.to(DEVICE)
    bbox_net.eval()

    inst_ds = []
    with torch.no_grad():
        for image_dict in tqdm(dataset):
            for frame in frames:
                img = faster_read_frame_at_index(image_dict['file_name'], frame)
                try:
                    inputs = {'image': torch.as_tensor(img.astype("float32").transpose(2, 0, 1)), 'height': image_dict['height'], 'width': image_dict['width']}
                except AttributeError:
                    print('Video {} not found (at frame {})'.format(image_dict['file_name'], frame))
                    continue
                outputs = bbox_net([inputs])[0]
                scores = outputs['instances'].scores
                pred_boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy().astype(int).tolist()

                for bbox, score in zip(pred_boxes, scores):
                    inst_dict = copy.deepcopy(image_dict)
                    inst_dict['bbox'] = bbox
                    inst_dict['score'] = float(score)
                    inst_dict['frame'] = frame
                    inst_dict['bbox_aspect'] = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
                    inst_dict['aspect_group'] = int(np.abs(inst_dict['bbox_aspect'] - aspect_template).argmin())

                    inst_ds.append(inst_dict)

    del bbox_net
    torch.cuda.empty_cache()

    return inst_ds

def get_pred_vid(dir):

    dataset = get_vid_dicts(dir)
    if DEBUG:
        dataset = dataset[:50]

    cfg = get_cfg()
    add_vovnet_config(cfg)
    cfg.merge_from_file('/myspace/one_cls_faster_rcnn_V_99_FPN/config.yaml')
    cfg.MODEL.WEIGHTS = '/myspace/one_cls_faster_rcnn_V_99_FPN/model_0099999.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95

    cfg.TEST.AUG.ENABLED = True
    cfg.TEST.AUG.MIN_SIZES = (608, )
    cfg.TEST.AUG.MAX_SIZE = 4000
    cfg.TEST.AUG.FLIP = False

    bbox_net = build_model(cfg)
    DetectionCheckpointer(bbox_net).load(cfg.MODEL.WEIGHTS)
    bbox_net = GeneralizedRCNNWithTTA(cfg, bbox_net)

    inst_ds = infer_vid(bbox_net, dataset, frames=list(range(80, 400, 20)))

    return inst_ds