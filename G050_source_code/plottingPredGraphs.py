import torch
import cv2
import argparse
import time
import numpy as np
import albumentations as A
import os
import matplotlib.pyplot as plt

from class_names import class_names
from model import build_model

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input video')
parser.add_argument('-c', '--clip-len', dest='clip_len', default=16, type=int,
                    help='number of frames to consider for each prediction')
args = parser.parse_args()

OUT_DIR = os.path.join('..', 'outputs', 'inference')
os.makedirs(OUT_DIR, exist_ok=True)

# Define the transforms.
transform = A.Compose([
    A.Resize(360, 640, always_apply=True),
    A.CenterCrop(360, 360, always_apply=True),
    A.Normalize(mean=[0.43216, 0.394666, 0.37645],
                std=[0.22803, 0.22145, 0.216989],
                always_apply=True)
])

# get the labels
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('../outputs/swin_360_best_model.pth', map_location=torch.device('cpu'))
# load the model
model = build_model(
    fine_tune=False,
    num_classes=len(class_names)
)
# load the model onto the computation device
model.load_state_dict(ckpt['model_state_dict'])
model = model.eval().to(device)

cap = cv2.VideoCapture(args.input)
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

save_name = f"{args.input.split('/')[-1].split('.')[0]}"
# define codec and create VideoWriter object
out = cv2.VideoWriter(f"{OUT_DIR}/{save_name}.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), fps,
                      (frame_width, frame_height))

frame_count = 0  # to count total frames
total_fps = 0  # to get the final frames per second
# a clips list to append and store the individual frames
clips = []

predictions = []  # to store all predictions along with their timestamps

# read until end of video
while (cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        # get the start time
        start_time = time.time()
        image = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(image=frame)['image']
        clips.append(frame)
        if len(clips) == args.clip_len:
            with torch.no_grad():  # we do not want to backprop any gradients
                input_frames = np.array(clips)
                # add an extra dimension
                input_frames = np.expand_dims(input_frames, axis=0)
                # transpose to get [1, 3, num_clips, height, width]
                input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))
                # convert the frames to tensor
                input_frames = torch.tensor(input_frames, dtype=torch.float32)
                input_frames = input_frames.to(device)
                # forward pass to get the predictions
                outputs = model(input_frames)
                # get the prediction index
                _, preds = torch.max(outputs.data, 1)

                # map predictions to the respective class names
                label = class_names[preds].strip()
                predictions.append((cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, preds.item()))  # Convert milliseconds to seconds
            # get the end time
            end_time = time.time()
            # get the fps
            fps = 1 / (end_time - start_time)
            # add fps to total fps
            total_fps += fps
            # increment frame count
            frame_count += 1
            wait_time = max(1, int(fps / 4))
            cv2.putText(image, label, (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)
            clips.pop(0)
            # cv2.imshow('image', image)
            # out.write(image)
            # press `q` to exit
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Plotting predictions
timestamps, preds = zip(*predictions)
timestamps = np.array(timestamps)
preds = np.array(preds)

# Converting predictions to binary (1 for risky, 0 for non-risky)
# Convert preds to strings indicating risky and non-risky
y_labels = ['risky' if p == 1 else 'non-risky' for p in preds]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(timestamps - timestamps[0], preds, 'r-')  # Use preds directly for plotting
plt.axhline(y=0.5, color='b', linestyle='--')
plt.yticks([0, 1], ['non-risky', 'risky'])  # Set y-axis ticks and labels
plt.xlabel('Time (seconds)')
plt.ylabel('Risk')
plt.title('Risk Prediction Over Time')
plt.savefig(f"boxing_vid_preds.png")
plt.show()

# binary_preds = np.where(preds == 1, 1, 0)

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(timestamps - timestamps[0], binary_preds, 'r-')
# plt.axhline(y=0.5, color='b', linestyle='--')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Risk')
# plt.title('Risk Prediction Over Time')
# plt.show()
