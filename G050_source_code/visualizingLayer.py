import time
import cv2
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import torch
import numpy as np
from model import build_model
import albumentations as A


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('/Users/rijul/Library/CloudStorage/OneDrive-UniversityofEdinburgh/MLPData/MLP_CW3/20230717_Training_a_Video_Classification_Model_from_Torchvision/src/swin_best_model.pth', map_location=device)

# load the model
model = build_model(
    fine_tune=False,
    num_classes=2
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Get the last ShiftedWindowAttention layer
last_attn_layer = model.features[0][-2].attn

# Hook to get the attention maps
attn_maps = None

def get_attn_maps(module, input, output):
    global attn_maps
    attn_maps = output

hook = last_attn_layer.register_forward_hook(get_attn_maps)

# Pass a dummy input through the model

# Define the transforms.
transform = A.Compose([
    A.Resize(360, 640, always_apply=True),
    A.CenterCrop(360, 360, always_apply=True),
    A.Normalize(mean = [0.43216, 0.394666, 0.37645],
                std = [0.22803, 0.22145, 0.216989], 
                always_apply=True)
])

cap = cv2.VideoCapture('/Users/rijul/Library/CloudStorage/OneDrive-UniversityofEdinburgh/MLPData/MLP_CW3/20230717_Training_a_Video_Classification_Model_from_Torchvision/src/merge_test_videos/testVideBox.mp4')
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# define codec and create VideoWriter object 


frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second
# a clips list to append and store the individual frames
clips = []

input_frames = None
# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        # get the start time
        start_time = time.time()
        image = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(image=frame)['image']
        clips.append(frame)
        if len(clips) == 16:
            with torch.no_grad(): # we do not want to backprop any gradients
                input_frames = np.array(clips)
                # add an extra dimension        
                input_frames = np.expand_dims(input_frames, axis=0)
                # transpose to get [1, 3, num_clips, height, width]
                input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))
                # convert the frames to tensor
                input_frames = torch.tensor(input_frames, dtype=torch.float32)
                input_frames = input_frames.to(device)
                break

output = model(input_frames)

# Remove the hook
hook.remove()

# Process the attention maps
attn_maps = attn_maps.squeeze(1)  # Remove the dummy batch dimension
num_windows = attn_maps.shape[1]  # Get the number of window partitions
attn_maps = attn_maps.view(-1, num_windows, attn_maps.shape[-2], attn_maps.shape[-1])  # Flatten window partitions
attn_maps = attn_maps.permute(0, 2, 3, 1)  # Reshape for visualization
attn_maps = attn_maps.cpu().numpy()  # Move to CPU and convert to numpy

# Visualize the attention maps
num_heads = attn_maps.shape[0]
print("Shape of attention maps: ",attn_maps.shape)

output_reshaped=attn_maps.reshape(90*90,96,8)

plt.figure(figsize=(12, 6))
plt.imshow(output_reshaped.mean(axis=0), cmap='viridis', aspect='auto')
plt.title('Multi-Head Attention Visualization')
plt.xlabel('Heads')
plt.ylabel('Positions')
plt.colorbar()
plt.show()
# fig, axs = plt.subplots(nrows=num_heads, ncols=1, figsize=(6, 20))

# for i, ax in enumerate(axs):
    
#     ax.imshow(attn_maps[i][0].reshape(24,32), cmap='gray')  # Average over window partitions
#     ax.set_title(f'Head {i+1}')
#     ax.axis('off')

# plt.tight_layout()
# plt.show()