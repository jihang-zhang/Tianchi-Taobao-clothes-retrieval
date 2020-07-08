import os
from os.path import sep, join
import json
from tqdm import tqdm

import cv2
import imagesize
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from logger import create_folder

DEBUG = False

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

# thing_cls = ['短袖上衣', '长袖上衣', '短袖衬衫', '长袖衬衫', '背心上衣', '吊带上衣', '无袖上衣', '短外套', '短马甲', '长袖连衣裙', '短袖连衣裙', '无袖连衣裙', '长马甲', '长外套', '连体衣', '古风', '短裙', '中等半身裙', '长半身裙', '短裤', '中裤', '长裤', '背带裤']
thing_cls = ['clothes']

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
        
        annotations = []
        for instance in d['annotations']:
            annotations.append({
                'bbox': instance['box'],
                'bbox_mode': BoxMode.XYXY_ABS,
#                 'category_id': label2cat[instance['label']]
                'category_id': 0 # just detect clothes
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
        
        annotations = []
        for instance in j['annotations']:
            annotations.append({
                'bbox': instance['box'],
                'bbox_mode': BoxMode.XYXY_ABS,
#                 'category_id': label2cat[instance['label']]
                'category_id': 0 # just detect clothes
            })
        image_dict['annotations'] = annotations
        result.append(image_dict)
    return result


def get_dicts(input_dir, type, n_items=None):
    if type == 'image':
        func = get_item_dicts_img
    elif type == 'video':
    	func = get_item_dicts_vid

    item_id_list = os.listdir(pjoin(input_dir, 'image_annotation'))
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
        get_dicts('/tcdata_train/train_dataset_part5', 'image')
    return result

def get_dicts_train_all_vid():
    result = \
        get_dicts('/tcdata_train/train_dataset_part1', 'video') + \
        get_dicts('/tcdata_train/train_dataset_part2', 'video') + \
        get_dicts('/tcdata_train/train_dataset_part3', 'video') + \
        get_dicts('/tcdata_train/train_dataset_part4', 'video') + \
        get_dicts('/tcdata_train/train_dataset_part5', 'video')
    return result

def get_dicts_train_all():
    return get_dicts_train_all_img() + get_dicts_train_all_vid()

def get_dicts_val_all_img():
    result = \
        get_dicts('/tcdata_train/validation_dataset_part1', 'image') + \
        get_dicts('/tcdata_train/validation_dataset_part2', 'image')
    return result

def get_dicts_val_all_vid():
    result = \
        get_dicts('/tcdata_train/validation_dataset_part1', 'video') + \
        get_dicts('/tcdata_train/validation_dataset_part2', 'video')
    return result

def get_dicts_train_mini_img():
    return get_dicts('/tcdata_train/train_dataset_part1', 'image', n_items=200)

def get_dicts_train_mini_vid():
    return get_dicts('/tcdata_train/train_dataset_part1', 'video', n_items=100)

def get_dicts_train_mini():
    return get_dicts_train_mini_img() + get_dicts_train_mini_vid()

def get_dicts_val_mini_img():
    return get_dicts('/tcdata_train/validation_dataset_part1', 'image', n_items=200)

def get_dicts_val_mini_vid():
    return get_dicts('/tcdata_train/validation_dataset_part1', 'video', n_items=100)

if DEBUG:
    DatasetCatalog.register('taobao_train_mini', get_dicts_train_mini)
    DatasetCatalog.register('taobao_val_mini_img', get_dicts_val_mini_img)
    DatasetCatalog.register('taobao_val_mini_vid', get_dicts_val_mini_vid)

    MetadataCatalog.get('taobao_train_mini').set(thing_classes=thing_cls)
    MetadataCatalog.get('taobao_val_mini_img').set(thing_classes=thing_cls)
    MetadataCatalog.get('taobao_val_mini_vid').set(thing_classes=thing_cls)
else:
    DatasetCatalog.register('taobao_train_all', get_dicts_train_all)
    DatasetCatalog.register('taobao_val_mini_img', get_dicts_val_mini_img)
    DatasetCatalog.register('taobao_val_mini_vid', get_dicts_val_mini_vid)

    MetadataCatalog.get('taobao_train_all').set(thing_classes=thing_cls)
    MetadataCatalog.get('taobao_val_mini_img').set(thing_classes=thing_cls)
    MetadataCatalog.get('taobao_val_mini_vid').set(thing_classes=thing_cls)

def main():
    taobao_train_img = get_dicts_train_all_img()
    taobao_train_vid = get_dicts_train_all_vid()

    taobao_val_img = get_dicts_val_all_img()
    taobao_val_vid = get_dicts_val_all_vid()

    create_folder('/myspace/input')

    with open('/myspace/input/taobao_train_img.json', 'w') as f:
        json.dump(taobao_train_img, f)

    with open('/myspace/input/taobao_val_img.json', 'w') as f:
        json.dump(taobao_val_img, f)

    with open('/myspace/input/taobao_train_vid.json', 'w') as f:
        json.dump(taobao_train_vid, f)

    with open('/myspace/input/taobao_val_vid.json', 'w') as f:
        json.dump(taobao_val_vid, f)

if __name__ == "__main__":
    main()