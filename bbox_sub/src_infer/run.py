import re
import json
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from tqdm import tqdm
from infer_img import get_pred_img
from infer_vid import get_pred_vid

# pred_img = get_pred_img('../../input/validation_dataset_part1/image')
# pred_vid = get_pred_vid('../../input/validation_dataset_part1/video')

pred_img = get_pred_img('/tcdata/test_dataset_fs/image')

with open('/myspace/test_img_bbox_16.json', 'w') as f:
    json.dump(pred_img, f)

pred_vid = get_pred_vid('/tcdata/test_dataset_fs/video')

with open('/myspace/test_vid_bbox_16.json', 'w') as f:
    json.dump(pred_vid, f)
