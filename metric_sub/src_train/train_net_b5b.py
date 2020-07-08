import time
import random
import os
import sys
import json

import logging
import functools
from collections import defaultdict
from tqdm import tqdm

import math
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from optimizer import Ranger
from transformers import get_cosine_schedule_with_warmup

from network import ENet
from loss import mixed_loss
from utils import normalize
from make_dataloader import make_dataloader
from logger import create_folder, create_logging

from torch.cuda.amp import autocast, GradScaler

# Configuration
class Config:
    def __init__(self):
        self.SEED = 45678
        self.EVAL_INTERVAL = 1
        self.MAX_EPOCHS = 10
        self.STAGE_EPOCHS = 10
        self.BATCH_SIZE = 32
        self.INFER_BATCH_SIZE = 32
        self.LOG_INTERVAL = 200
        self.SCALE = 'SM'
        self.OUTPUT_PATH = '/myspace/output/arcface_b5b'
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.LOGS_DIR = os.path.join(self.OUTPUT_PATH, 'logs')
        self.DEBUG = False

        self.WARMUP = 0.025
        self.LR = 0.00035
        self.GRAD_UPDATE_STEP = 1
        self.XBM = None

        self.BACKBONE = 'tf_efficientnet_b5_ns'
        self.COS_LAYER = True
        self.M = 0.30
        self.FEAT_DIM = 512
        self.N_ID = 47652
        self.DROPOUT = 0.
        self.POOLING = 'gem_freeze'

        self.RESUME = False
        self.PRETRAINED_FILE = None


def do_train(cfg, net, criterion, optimizer, scheduler, train_ldr, epoch, global_step):

    start_time = time.time()

    total_loss_a = 0

    tq = enumerate(train_ldr)

    net.train()
    scaler = GradScaler()
    optimizer.zero_grad()
    for batch_idx, (image, label_a, _) in tq:

        image = image.to(cfg.DEVICE)
        label_a = label_a.to(cfg.DEVICE)

        with autocast():
            if cfg.XBM is not None:
                score, label_a, _ = net(image, label_a)
            else:
                score, _ = net(image, label_a)
            loss_a = criterion(score, label_a)
            loss = loss_a

        scaler.scale(loss).backward()

        if (batch_idx + 1) % cfg.GRAD_UPDATE_STEP == 0:             # Wait for several backward steps
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            optimizer.zero_grad()

            global_step += 1

        total_loss_a += loss_a.item()

        if batch_idx % cfg.LOG_INTERVAL == 0 and batch_idx > 0:
            cur_loss_a   = total_loss_a / cfg.LOG_INTERVAL
            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | {:3d}/{:3d} batches | ms/batch {:5.2f} | loss_a {:5.4f} |'.format(
                epoch, batch_idx, len(train_ldr), elapsed * 1000 / cfg.LOG_INTERVAL, cur_loss_a))
            
            total_loss_a = 0
            start_time = time.time()
    return global_step, scheduler.get_last_lr()[0]


def do_evaluate(cfg, net, valid_img_json_list, valid_vid_json_list, valid_img_ldr, valid_vid_ldr):

    batch_size = cfg.INFER_BATCH_SIZE

    net.eval()

    img_feats = defaultdict(list)
    vid_feats = defaultdict(list)

    # preds_img = np.zeros((len(valid_img_json_list), 3))
    # truths_img = np.zeros((len(valid_img_json_list), 3))

    # preds_vid = np.zeros((len(valid_vid_json_list), 3))
    # truths_vid = np.zeros((len(valid_vid_json_list), 3))

    tq_img = valid_img_ldr
    tq_vid = valid_vid_ldr

    with torch.no_grad():
        for batch_idx, (image, label_a, _) in enumerate(tq_img):
            image = image.to(cfg.DEVICE)
            feat = net(image)
            feat = F.normalize(feat)
            for idx, inst_id in enumerate(label_a):
                img_feats[int(inst_id)].append(feat[idx].detach().cpu().numpy())

        img_id_feats = {k: normalize(np.stack(v).mean(axis=0)) for k, v in img_feats.items()}

        for batch_idx, (image, label_a, _) in enumerate(tq_vid):
            image = image.to(cfg.DEVICE)
            feat = net(image)
            feat = F.normalize(feat)
            for idx, inst_id in enumerate(label_a):
                vid_feats[int(inst_id)].append(feat[idx].detach().cpu().numpy())

        vid_id_feats = {k: normalize(np.stack(v).mean(axis=0)) for k, v in vid_feats.items()}

    img_id_feats = pd.DataFrame.from_dict(img_id_feats, orient='index')

    vid_id_feats = pd.DataFrame.from_dict(vid_id_feats, orient='index')
    # vid_id_feats = vid_id_feats[vid_id_feats.index.isin(img_id_feats.index)]

    cos_map = vid_id_feats.values.dot(img_id_feats.values.T)
    preds_idx = cos_map.argmax(1)
    preds_img_inst = img_id_feats.index.values[preds_idx]

    accuracy = (vid_id_feats.index.values == preds_img_inst).mean()

    return accuracy


def main(cfg):
    create_logging(cfg.LOGS_DIR, 'w')

    with open("/myspace/input/taobao_round1_img_inst_80.json", "r") as f:
        train_img_json_list = json.load(f)

    with open("/myspace/input/taobao_round1_vid_inst_80.json", "r") as f:
        train_vid_json_list = json.load(f)

    train_json_list = train_img_json_list + train_vid_json_list
    if cfg.DEBUG:
        train_json_list = train_json_list[:100] + train_json_list[-100:]
    train_ldr = make_dataloader(train_json_list, cfg.BATCH_SIZE, cfg.SCALE, 'train')

    with open("/myspace/input/taobao_round2_img_inst_80_id3000.json", "r") as f:
        valid_img_json_list = json.load(f)

    with open("/myspace/input/taobao_round2_vid_inst_80_id3000.json", "r") as f:
        valid_vid_json_list = json.load(f)

    if cfg.DEBUG:
        valid_img_json_list = valid_img_json_list[:100]
        valid_vid_json_list = valid_vid_json_list[:100]
    valid_img_ldr = make_dataloader(valid_img_json_list, cfg.INFER_BATCH_SIZE, cfg.SCALE, 'valid')
    valid_vid_ldr = make_dataloader(valid_vid_json_list, cfg.INFER_BATCH_SIZE, cfg.SCALE, 'valid')

    num_train_optimization_steps = int(cfg.MAX_EPOCHS * len(train_json_list) / cfg.BATCH_SIZE / cfg.GRAD_UPDATE_STEP)
    num_warmup_steps = int(num_train_optimization_steps * cfg.WARMUP)

    net = ENet(num_classes=cfg.N_ID, feat_dim=cfg.FEAT_DIM, cos_layer=cfg.COS_LAYER, 
        xbm=cfg.XBM, dropout=cfg.DROPOUT, m=cfg.M, pool=cfg.POOLING, image_net=cfg.BACKBONE, pretrained=False).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    # optimizer = Ranger(net.parameters(), lr=cfg.LR)
    optimizer = optim.Adam(net.parameters(), lr=cfg.LR)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_train_optimization_steps)

    if cfg.RESUME:
        checkpoint = torch.load(cfg.PRETRAINED_FILE)
        global_step = checkpoint['global_step']
        best_so_far = checkpoint['score']
        current_epoch = checkpoint['epoch'] + 1

        logging.info("Resuming from epoch {}...".format(checkpoint['epoch']))

        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['schedlue_state_dict'])

        del checkpoint

    else:
        checkpoint = torch.load(cfg.PRETRAINED_FILE)
        global_step = 0
        best_so_far = 0
        current_epoch = 0

        logging.info("Loading pretrained model from ImageNet pretrained")
        net.load_state_dict(checkpoint['model_state_dict'])

        del checkpoint

    for epoch in tqdm(range(current_epoch, current_epoch + cfg.STAGE_EPOCHS)):

        epoch_start_time = time.time()
        logging.info('Start training')

        global_step, current_lr = do_train(cfg, net, criterion, optimizer, scheduler, train_ldr, epoch, global_step)
        logging.info("Current Learning rate: {:.5f}".format(current_lr))

        if (epoch % cfg.EVAL_INTERVAL) == (cfg.EVAL_INTERVAL-1):
            val_score = do_evaluate(cfg, net, valid_img_json_list, valid_vid_json_list, valid_img_ldr, valid_vid_ldr)

            logging.info('-' * 89)
            logging.info('end of epoch {:4d} | time: {:5.2f}s     | match acc {:5.4f}    |'.format(
                epoch, (time.time() - epoch_start_time), val_score))

            logging.info('-' * 89)

            if val_score > best_so_far:
                best_so_far = val_score
            logging.info('Saving epoch {} model...'.format(epoch))
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'schedlue_state_dict': scheduler.state_dict(),
                'score': val_score,
            }, os.path.join(cfg.OUTPUT_PATH, '{}.pt'.format(epoch)))

    logging.info('Finish training - saving last checkpoint')
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'schedlue_state_dict': scheduler.state_dict(),
        'score': val_score,
    }, os.path.join(cfg.OUTPUT_PATH, 'final.pt'))

if __name__ == "__main__":
    # Check directory
    for dirname, _, filenames in os.walk('/myspace'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    cfg = Config()
    cfg.PRETRAINED_FILE = '../pretrained/effnet-b5_imagenet_pretrained.pt'

    main(cfg)
