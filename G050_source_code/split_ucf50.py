import os
import glob
import shutil
import numpy as np
import random

from tqdm.auto import tqdm

seed = 42
np.random.seed(seed)
random.seed(seed)

VALID_SPLIT = 0.10
TRAIN_SPLIT = 1.0 - (VALID_SPLIT)
TRAINING_PERCENTAGE = 0.35

ROOT_DIR = os.path.join('..', 'input','UCF50')
DST_ROOT = os.path.join('..', 'input', 'ucf50_train_valid')
os.makedirs(DST_ROOT, exist_ok=True)

all_videos = glob.glob(os.path.join(ROOT_DIR, '*', '*'), recursive=True)
all_videos.sort()
random.shuffle(all_videos)

total_samples = len(all_videos)

def copy_data(video_list, split='train'):
    for i, video_name in tqdm(enumerate(video_list), total=len(video_list)):
        class_name = video_name.split(os.path.sep)[-2]
        data_dir = os.path.join(DST_ROOT, split, class_name)
        file_name = video_name.split(os.path.sep)[-1]
        os.makedirs(os.path.join(data_dir), exist_ok=True)
        shutil.copy(
            os.path.join(video_name),
            os.path.join(data_dir, file_name)
        )

train_videos = all_videos[0:int(total_samples*TRAIN_SPLIT)]
valid_videos = all_videos[
    int(total_samples*TRAIN_SPLIT):int(total_samples*TRAIN_SPLIT+total_samples*VALID_SPLIT)
]

copy_data(train_videos[:int(TRAINING_PERCENTAGE*len(train_videos))], 'train')
copy_data(valid_videos, 'valid')