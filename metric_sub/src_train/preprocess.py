import os
from os.path import sep, join, splitext
import json
import copy
from tqdm import tqdm

import numpy as np
import pandas as pd

import cv2
import imagesize

from logger import create_folder

DEBUG = False
aspect_template = np.array([0.25, 0.335, 0.415, 0.5, 0.5721925, 0.66857143, 0.8, 1., 1.25, 1.4957264, 1.74766355, 2.])

def pjoin(*args, **kwargs):
    return join(*args, **kwargs).replace(sep, '/')

label2cat = {
    '短袖上衣': 0,
    '长袖上衣': 1,
    '短袖衬衫': 2,
    '长袖衬衫': 3,
    '背心上衣': 4,
    '吊带上衣': 5,
    '无袖上衣': 6,
    '短外套': 7,
    '短马甲': 8,
    '长袖连衣裙':  9,
    '短袖连衣裙': 10,
    '无袖连衣裙': 11,
    '长马甲': 12,
    '长外套': 13,
    '连体衣': 14,
    '古风': 15,
    '古装': 15,
    '短裙': 16,
    '中等半身裙': 17,
    '长半身裙': 18,
    '短裤': 19,
    '中裤': 20,
    '长裤': 21,
    '背带裤': 22
}

def get_item_dicts_img(input_dir, item_id):
    image_dir = pjoin(input_dir, 'image', item_id)
    
    json_dir = pjoin(input_dir, 'image_annotation', item_id)
    json_list = os.listdir(json_dir)
    
    result = []
    for j in json_list:
        with open(pjoin(json_dir, j)) as f:
            d = json.load(f)
        
        image_dict = {}
        image_dict['file_name'] = pjoin(image_dir, d['img_name'])
        w, h = imagesize.get(image_dict['file_name'])
        image_dict['height'] = h
        image_dict['width'] = w
        image_dict['image_id'] = d['item_id'] + '_' + d['img_name']
        image_dict['type'] = 'image'
        image_dict['item_id'] = d['item_id']
        
        annotations = []
        for instance in d['annotations']:
            annotations.append({
                'display': instance['display'],
                'bbox': instance['box'],
                'category_id': label2cat[instance['label']],
                'viewpoint': instance['viewpoint'],
                'instance_id': instance['instance_id'],
                'bbox_aspect': (instance['box'][2] - instance['box'][0]) / (instance['box'][3] - instance['box'][1])
            })
        image_dict['annotations'] = annotations
        result.append(image_dict)
    return result


def get_item_dicts_vid(input_dir, item_id):
    image_dir = pjoin(input_dir, 'video', item_id+'.mp4')
    json_dir = pjoin(input_dir, 'video_annotation', item_id+'.json')
    
    with open(json_dir) as f:
        d = json.load(f)

    capture = cv2.VideoCapture(image_dir)
    w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    result = []
    for j in d['frames']:
        
        image_dict = {}
        image_dict['file_name'] = image_dir
        image_dict['height'] = h
        image_dict['width'] = w
        image_dict['image_id'] = d['video_id'] + '_' + str(j['frame_index'])
        image_dict['frame'] = j['frame_index']
        image_dict['type'] = 'video'
        image_dict['item_id'] = d['video_id']
        
        annotations = []
        for instance in j['annotations']:
            annotations.append({
                'display': instance['display'],
                'bbox': instance['box'],
                'category_id': label2cat[instance['label']],
                'viewpoint': instance['viewpoint'],
                'instance_id': instance['instance_id'],
                'bbox_aspect': (instance['box'][2] - instance['box'][0]) / (instance['box'][3] - instance['box'][1])
            })
        image_dict['annotations'] = annotations
        result.append(image_dict)
    return result


def get_dicts(input_dir, type, n_items=None):
    if type == 'image':
        func = get_item_dicts_img
        item_id_list = os.listdir(pjoin(input_dir, 'image_annotation'))
    elif type == 'video':
        func = get_item_dicts_vid
        item_id_list = [splitext(x)[0] for x in os.listdir(pjoin(input_dir, 'video'))]

    if n_items is not None:
        item_id_list = item_id_list[:n_items]
    
    result = []
    for item_id in tqdm(item_id_list):
        result += func(input_dir, item_id)
    
    return result

def get_dicts_train_all_img():
    result = \
        get_dicts('/tcdata_train/train_dataset_part1', 'image') + \
        get_dicts('/tcdata_train/train_dataset_part2', 'image') + \
        get_dicts('/tcdata_train/train_dataset_part3', 'image') + \
        get_dicts('/tcdata_train/train_dataset_part4', 'image') + \
        get_dicts('/tcdata_train/train_dataset_part5', 'image') + \
        get_dicts('/tcdata_train/train_dataset_part6', 'image') + \
        get_dicts('/tcdata_train/validation_dataset_part1', 'image') + \
        get_dicts('/tcdata_train/validation_dataset_part2', 'image')
    return result

def get_dicts_train_all_vid():
    result = \
        get_dicts('/tcdata_train/train_dataset_part1', 'video') + \
        get_dicts('/tcdata_train/train_dataset_part2', 'video') + \
        get_dicts('/tcdata_train/train_dataset_part3', 'video') + \
        get_dicts('/tcdata_train/train_dataset_part4', 'video') + \
        get_dicts('/tcdata_train/train_dataset_part5', 'video') + \
        get_dicts('/tcdata_train/train_dataset_part6', 'video') + \
        get_dicts('/tcdata_train/validation_dataset_part1', 'video') + \
        get_dicts('/tcdata_train/validation_dataset_part2', 'video')
    return result

def get_dicts_val_all_img():
    result = \
        get_dicts('/tcdata_train/validation_dataset_part3', 'image') + \
        get_dicts('/tcdata_train/validation_dataset_part4', 'image')
    return result

def get_dicts_val_all_vid():
    result = \
        get_dicts('/tcdata_train/validation_dataset_part3', 'video') + \
        get_dicts('/tcdata_train/validation_dataset_part4', 'video')
    return result


# instance json helpers
def get_instance_dict_img(metric_json):
    result = []
    for d in tqdm(metric_json):
        img_dict = {k: v for k, v in d.items() if k in ('file_name', 'height', 'width', 'image_id', 'type', 'item_id')}
        for ins in d['annotations']:
            if ins['instance_id'] == 0:
                continue
            ins_dict = copy.deepcopy(img_dict)
            ins_dict.update(ins)
            ins_dict['aspect_group'] = int(np.abs(ins_dict['bbox_aspect'] - aspect_template).argmin())
            result.append(ins_dict)
    return result

def get_instance_dict_vid(metric_json):
    result = []
    for d in tqdm(metric_json):
        if d['frame'] < 80:
            continue
        img_dict = {k: v for k, v in d.items() if k in ('file_name', 'height', 'width', 'image_id', 'frame', 'type', 'item_id')}
        for ins in d['annotations']:
            if ins['instance_id'] == 0:
                continue
            ins_dict = copy.deepcopy(img_dict)
            ins_dict.update(ins)
            ins_dict['aspect_group'] = int(np.abs(ins_dict['bbox_aspect'] - aspect_template).argmin())
            result.append(ins_dict)
    return result


def main():
    taobao_train_img = get_dicts_train_all_img()
    taobao_train_vid = get_dicts_train_all_vid()

    taobao_val_img = get_dicts_val_all_img()
    taobao_val_vid = get_dicts_val_all_vid()

    create_folder('/myspace/input')

    with open('/myspace/input/taobao_train_img_metric.json', 'w') as f:
        json.dump(taobao_train_img, f)

    with open('/myspace/input/taobao_val_img_metric.json', 'w') as f:
        json.dump(taobao_val_img, f)

    with open('/myspace/input/taobao_train_vid_metric.json', 'w') as f:
        json.dump(taobao_train_vid, f)

    with open('/myspace/input/taobao_val_vid_metric.json', 'w') as f:
        json.dump(taobao_val_vid, f)

    # with open('/myspace/input/taobao_train_img_metric.json', 'r') as f:
    #     taobao_train_img = json.load(f)

    # with open('/myspace/input/taobao_val_img_metric.json', 'r') as f:
    #     taobao_val_img = json.load(f)

    # with open('/myspace/input/taobao_train_vid_metric.json', 'r') as f:
    #     taobao_train_vid = json.load(f)

    # with open('/myspace/input/taobao_val_vid_metric.json', 'r') as f:
    #     taobao_val_vid = json.load(f)


    train_img_inst = get_instance_dict_img(taobao_train_img)
    train_vid_inst = get_instance_dict_vid(taobao_train_vid)

    train_img_inst_df = pd.DataFrame.from_dict(train_img_inst)
    train_vid_inst_df = pd.DataFrame.from_dict(train_vid_inst)

    img_inst_ids = train_img_inst_df['instance_id'].unique()
    print('No. instances in round 1 images: {}'.format(len(img_inst_ids)))
    vid_inst_ids = train_vid_inst_df['instance_id'].unique()
    print('No. instances in round 1 videos: {}'.format(len(vid_inst_ids)))
    inst_ids = np.union1d(img_inst_ids, vid_inst_ids)
    print('Total no. unique instances in round 1: {}'.format(len(inst_ids)))

    # encode instance ids
    instance_encoder = {inst_id: i for i, inst_id in enumerate(inst_ids)}
    train_img_inst_df['instance_id'] = train_img_inst_df['instance_id'].map(instance_encoder)
    train_vid_inst_df['instance_id'] = train_vid_inst_df['instance_id'].map(instance_encoder)

    train_img_inst = train_img_inst_df.to_dict(orient='records')
    train_vid_inst = train_vid_inst_df.to_dict(orient='records')

    with open('/myspace/input/taobao_round1_img_inst_80.json', 'w') as f:
        json.dump(train_img_inst, f)

    with open('/myspace/input/taobao_round1_vid_inst_80.json', 'w') as f:
        json.dump(train_vid_inst, f)

    # save encoder map
    instance_encoder = pd.DataFrame.from_dict(instance_encoder, orient='index').reset_index()
    instance_encoder.columns = ['instance_id', 'instance_cat']
    instance_encoder.to_feather('/myspace/input/taobao_round1_instance_encoder_80.feather')


    # round 2 data
    val_img_inst = get_instance_dict_img(taobao_val_img)
    val_vid_inst = get_instance_dict_vid(taobao_val_vid)

    with open('/myspace/input/taobao_round2_img_inst_80.json', 'w') as f:
        json.dump(val_img_inst, f)
        
    with open('/myspace/input/taobao_round2_vid_inst_80.json', 'w') as f:
        json.dump(val_vid_inst, f)

    val_img_inst_df = pd.DataFrame.from_dict(val_img_inst)
    val_vid_inst_df = pd.DataFrame.from_dict(val_vid_inst)

    print('No. instances in round 2 images: {}'.format(val_img_inst_df['instance_id'].nunique()))
    print('No. instances in round 2 videos: {}'.format(val_vid_inst_df['instance_id'].nunique()))

    shared_inst = np.intersect1d(val_img_inst_df['instance_id'], val_vid_inst_df['instance_id'])
    print('No. shared instances in round 2: {}'.format(len(shared_inst)))

    img_sub_df = val_img_inst_df[val_img_inst_df['instance_id'].isin(shared_inst[:3000])]
    vid_sub_df = val_vid_inst_df[val_vid_inst_df['instance_id'].isin(shared_inst[:3000])]

    with open('/myspace/input/taobao_round2_img_inst_80_id3000.json', 'w') as f:
        json.dump(img_sub_df.to_dict('records'), f)

    with open('/myspace/input/taobao_round2_vid_inst_80_id3000.json', 'w') as f:
        json.dump(vid_sub_df.to_dict('records'), f)

if __name__ == "__main__":
    main()