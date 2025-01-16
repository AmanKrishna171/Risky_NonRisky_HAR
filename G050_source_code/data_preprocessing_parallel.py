import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import cv2
import time
import utils
import numpy as np
import re
import pickle
from tqdm import tqdm
import concurrent.futures

NUM_FRAMES_PER_CLIP = 16

video_dir = 'input'
video_files = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
labels = [f.split('_')[-1].split('.')[0] for f in video_files]

try:
    with open('processed_videos.pkl', 'rb') as f:
        processed_videos = pickle.load(f)
except FileNotFoundError:
    processed_videos = []

def process_video(idx):
    video_path = os.path.join(video_dir, video_files[idx])
    cap=cv2.VideoCapture(video_path)

    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')
        return

    if video_files[idx] in processed_videos:
        return

    processed_videos.append(video_files[idx])

    clips = []
    file_counter=0
    label=labels[idx]

    # read until end of video
    while(cap.isOpened()):
        # capture each frame of the video
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = utils.transform(image=frame)['image']
            clips.append(frame)
            if len(clips) == NUM_FRAMES_PER_CLIP:                    
                input_frames = np.array(clips)
                # add an extra dimension        
                input_frames = np.expand_dims(input_frames, axis=0)
                # transpose to get [1, 3, num_clips, height, width]
                input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))
                # convert the frames to tensor
                input_frames = torch.tensor(input_frames, dtype=torch.float32)
                # input_frames = input_frames.to(device)
                input_frames=input_frames.squeeze(0)

                if not os.path.exists('tensor_data'):
                    os.makedirs('tensor_data')

                # save the tensor
                filename = f"tensor_data/{label}_{file_counter}.pt"
                while os.path.exists(filename):
                    file_counter += 1
                    filename = f"tensor_data/{label}_{file_counter}.pt"
                torch.save(input_frames, filename)
                
                clips=clips[5:]
        else:
            break

    with open('processed_videos.pkl', 'wb') as f:
        pickle.dump(processed_videos, f)

# Use a ThreadPoolExecutor to process the videos in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(process_video, range(len(video_files))), total=len(video_files)))