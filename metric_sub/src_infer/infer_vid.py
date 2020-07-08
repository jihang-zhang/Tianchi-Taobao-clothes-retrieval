import json
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from network import ENet
from prepare_json import get_vid_dicts
from read_video import faster_read_frame_at_index
from make_dataloader import make_dataloader
from dataset import inst_process

DEBUG = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def infer_vid(metric_net, bbox_json, bbox_scale='SM', key='feat'):
    with open(bbox_json, 'r') as f:
        inst_ds = json.load(f)

    # infer metric
    metric_net.to(DEVICE)
    metric_net.eval()

    metric_img_ldr = tqdm(make_dataloader(inst_ds, 32, 'SM'))

    feat_list = []
    idx_list = []
    with torch.no_grad():
        for inst, idx in metric_img_ldr:
            feat = F.normalize(metric_net(inst)).detach().cpu().numpy().tolist()
            feat_list += feat
            idx_list += idx

    del metric_net
    torch.cuda.empty_cache()

    for f, i in zip(feat_list, idx_list):
        inst_ds[i][key] = f

    return inst_ds

def get_pred_vid(metric_net, bbox_json, save_path, key):

    # metric_net = ENet(num_classes=47652, feat_dim=512, cos_layer=True, xbm=512, dropout=0., m=0.30, image_net='tf_efficientnet_b7_ns', pretrained=False)
    # checkpoint = torch.load('/myspace/output/arcface_b7/final.pt')
    # metric_net.load_state_dict(checkpoint['model_state_dict'])

    inst_ds = infer_vid(metric_net, bbox_json, bbox_scale='SM', key=key)

    with open(save_path, 'w') as f:
        json.dump(inst_ds, f)

    return inst_ds