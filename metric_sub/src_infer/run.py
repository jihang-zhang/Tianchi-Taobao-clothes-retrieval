import os
import re
import gc
import json
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize

from tqdm import tqdm
from network import ENet
from infer_img import get_pred_img
from infer_vid import get_pred_vid

from logger import create_folder
from re_ranking import compute_distmat_using_gpu, re_ranking

# pred_img = get_pred_img('/myspace/test_img_bbox_16.json', '/myspace/output/arcface_b7/pred_img.json')
# pred_vid = get_pred_vid('/myspace/test_vid_bbox_16.json', '/myspace/output/arcface_b7/pred_vid.json')
# pred_img = get_pred_img('../../input/validation_dataset_part1/image')
# pred_vid = get_pred_vid('../../input/validation_dataset_part1/video')

# with open('/myspace/output/ensemble/pred_img_test_b5_r1_b6_b7.json', 'r') as f:
#     pred_img = json.load(f)
    
# with open('/myspace/output/ensemble/pred_vid_test_b5_r1_b6_b7.json', 'r') as f:
#     pred_vid = json.load(f)

create_folder('/myspace/output/ensemble')

# effnet-b5a predict
print('Predicting effnet-b5a features on test set...')
metric_net_b5a = ENet(num_classes=47652, feat_dim=512, cos_layer=True, xbm=None, dropout=0., m=0.30, pool='gem_freeze', image_net='tf_efficientnet_b5_ns', pretrained=False)
checkpoint = torch.load('/myspace/output/arcface_b5a/final.pt')
metric_net_b5a.load_state_dict(checkpoint['model_state_dict'])

pred_img = get_pred_img(metric_net_b5a, '/myspace/test_img_bbox_16.json', '/myspace/output/ensemble/pred_img_test_b5a.json', 'feat_b5a')
pred_vid = get_pred_vid(metric_net_b5a, '/myspace/test_img_bbox_16.json', '/myspace/output/ensemble/pred_vid_test_b5a.json', 'feat_b5a')

del metric_net_b5a, checkpoint, pred_img, pred_vid
gc.collect()

# effnet-b5b predict
print('Predicting effnet-b5b features on test set...')
metric_net_b5b = ENet(num_classes=47652, feat_dim=512, cos_layer=True, xbm=None, dropout=0., m=0.30, pool='gem_freeze', image_net='tf_efficientnet_b5_ns', pretrained=False)
checkpoint = torch.load('/myspace/output/arcface_b5b/final.pt')
metric_net_b5b.load_state_dict(checkpoint['model_state_dict'])

pred_img = get_pred_img(metric_net_b5b, '/myspace/output/ensemble/pred_img_test_b5a.json', '/myspace/output/ensemble/pred_img_test_b5a_b5b.json', 'feat_b5b')
pred_vid = get_pred_vid(metric_net_b5b, '/myspace/output/ensemble/pred_vid_test_b5a.json', '/myspace/output/ensemble/pred_vid_test_b5a_b5b.json', 'feat_b5b')

del metric_net_b5b, checkpoint, pred_img, pred_vid
gc.collect()

# effnet-b6 predict
print('Predicting effnet-b6 features on test set...')
metric_net_b6 = ENet(num_classes=47652, feat_dim=512, cos_layer=True, xbm=256, dropout=0., m=0.30, pool='gem', image_net='tf_efficientnet_b6_ns', pretrained=False)
checkpoint = torch.load('/myspace/output/arcface_b6/final.pt')
metric_net_b6.load_state_dict(checkpoint['model_state_dict'])

pred_img = get_pred_img(metric_net_b6, '/myspace/output/ensemble/pred_img_test_b5a_b5b.json', '/myspace/output/ensemble/pred_img_test_b5a_b5b_b6.json', 'feat_b6')
pred_vid = get_pred_vid(metric_net_b6, '/myspace/output/ensemble/pred_vid_test_b5a_b5b.json', '/myspace/output/ensemble/pred_img_test_b5a_b5b_b6.json', 'feat_b6')

del metric_net_b6, checkpoint, pred_img, pred_vid
gc.collect()


# effnet-b7 predict
print('Predicting effnet-b7 features on test set...')
metric_net_b7 = ENet(num_classes=47652, feat_dim=512, cos_layer=True, xbm=512, dropout=0., m=0.30, pool='gem_freeze', image_net='tf_efficientnet_b7_ns', pretrained=False)
checkpoint = torch.load('/myspace/output/arcface_b7/final.pt')
metric_net_b7.load_state_dict(checkpoint['model_state_dict'])

pred_img = get_pred_img(metric_net_b7, '/myspace/output/ensemble/pred_img_test_b5a_b5b_b6.json', '/myspace/output/ensemble/pred_img_test_b5a_b5b_b6_b7.json', 'feat_b7')
pred_vid = get_pred_vid(metric_net_b7, '/myspace/output/ensemble/pred_img_test_b5a_b5b_b6.json', '/myspace/output/ensemble/pred_img_test_b5a_b5b_b6_b7.json', 'feat_b7')

del metric_net_b7, checkpoint
gc.collect()


pred_img_df = pd.DataFrame.from_dict(pred_img)
pred_vid_df = pd.DataFrame.from_dict(pred_vid)

pred_img_df['feat'] = normalize(np.stack(pred_img_df['feat_b5a'] + pred_img_df['feat_b5b'] + pred_img_df['feat_b6'] + pred_img_df['feat_b7'])).tolist()
pred_img_df = pred_img_df.drop(columns=['feat_b5a', 'feat_b5b', 'feat_b6', 'feat_b7'])
pred_vid_df['feat'] = normalize(np.stack(pred_vid_df['feat_b5a'] + pred_vid_df['feat_b5b'] + pred_vid_df['feat_b6'] + pred_vid_df['feat_b7'])).tolist()
pred_vid_df = pred_vid_df.drop(columns=['feat_b5a', 'feat_b5b', 'feat_b6', 'feat_b7'])

gallary_feats = torch.tensor(pred_img_df['feat']).cuda()
query_feats = torch.tensor(pred_vid_df['feat']).cuda()

distmat = compute_distmat_using_gpu(gallary_feats, query_feats, mini_batch=1000)
torch.cuda.empty_cache()

distmat1 = distmat[:gallary_feats.size(0), gallary_feats.size(0):]
retrieve_indeces = distmat1.argmin(1)
retrieve_dists = distmat1.min(1)
retrieve_item_ids = pred_vid_df.loc[retrieve_indeces, 'item_id'].values

pred_img_df['retri_query_idx'] = retrieve_indeces
pred_img_df['retri_dist'] = retrieve_dists
pred_img_df['retri_item_id'] = retrieve_item_ids

del distmat, distmat1

def get_score(pred_img_df, pred_vid_df, s1, s2):

    gallary_grouped = pred_img_df.groupby('item_id')
    
    gallary_list = []
    frame_list = []
    query_list = []
    dist_list = []

    image_names = []
    item_boxes = []
    frame_boxes = []
    
    for item_id, _df in tqdm(gallary_grouped):
        
        retrieve_indeces = _df['retri_query_idx'].values
        retrieve_dist = _df['retri_dist'].values
        retrieve_item_ids = _df['retri_item_id'].values
        
        thr_s1 = np.quantile(retrieve_dist, s1)
        retrieved_cand = retrieve_item_ids[retrieve_dist < thr_s1]
        retrieved_cand_unique, retrieved_cand_cnt = np.unique(retrieved_cand, return_counts=True)
        retrieved_cand_unique = retrieved_cand_unique[retrieved_cand_cnt >= round(len(retrieved_cand) * s2)]
        if len(retrieved_cand_unique) == 0:
            continue
        elif len(retrieved_cand_unique) == 1:
            retrieved = retrieved_cand_unique[0]
        else:
            s = 0
            for r in retrieved_cand_unique:
                q = retrieve_item_ids == r
                m = retrieve_dist[q].mean()
                if m > s:
                    s = m
                    retrieved = r
                    
        # get image name & box
        retrieved_df = _df[retrieve_dist < thr_s1][retrieved_cand == retrieved]
        max_score_img_idx = retrieved_df['score'].values.argmax()
        image_name = re.search('/image/{}/(.*).jpg'.format(item_id),
                               retrieved_df['file_name'].values[max_score_img_idx]).group(1)
        item_box = retrieved_df['bbox'].values[max_score_img_idx]
        
        # get distance score
        retrieved_dist = retrieve_dist[retrieve_dist < thr_s1][retrieved_cand == retrieved].mean()
        
        # get frame & frame_box
        query_indeces = retrieve_indeces[retrieve_dist < thr_s1][retrieved_cand == retrieved]
        query_df = pred_vid_df.loc[query_indeces]
        max_score_vid_idx = query_df['score'].values.argmax()
        frame = query_df['frame'].values[max_score_vid_idx]
        frame_box = query_df['bbox'].values[max_score_vid_idx]
        
        gallary_list.append(item_id)
        frame_list.append(frame)
        query_list.append(retrieved)
        dist_list.append(retrieved_dist)

        image_names.append(image_name)
        item_boxes.append(item_box)
        frame_boxes.append(frame_box)
        
    result_df = pd.DataFrame({'query': query_list, 
                              'item_id': gallary_list,
                              'frame_index': frame_list,
                              'img_name': image_names,
                              'item_box': item_boxes,
                              'frame_box': frame_boxes,
                              'distance': dist_list
                             })
    
    return result_df

result_df = get_score(pred_img_df, pred_vid_df, 0.28, 0.5)
result_df = result_df.sort_values(by=['query', 'distance']).drop_duplicates(subset=['query'], ignore_index=True)

final_dict = {}
for k, v in result_df.set_index('query').to_dict(orient='index').items():
    if v['distance'] >= 1.12:
        continue

    final_dict[k] = {
        'item_id': v['item_id'],
        'frame_index': v['frame_index'],
        'result': [
            {
                'img_name': v['img_name'],
                'item_box': v['item_box'],
                'frame_box': v['frame_box']
            }
        ]
    }

with open('../result.json', 'w', encoding='utf-8') as f:
    json.dump(final_dict, f)

# Check directory
for dirname, _, filenames in os.walk('/myspace'):
    for filename in filenames:
        print(os.path.join(dirname, filename))