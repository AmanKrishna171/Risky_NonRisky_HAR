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

# Define the transforms.
transform = A.Compose([
    A.Resize(360, 640, always_apply=True),
    A.CenterCrop(360, 360, always_apply=True),
    A.Normalize(mean=[0.43216, 0.394666, 0.37645],
                std=[0.22803, 0.22145, 0.216989],
                always_apply=True)
])

cap = cv2.VideoCapture('/Users/rijul/Library/CloudStorage/OneDrive-UniversityofEdinburgh/MLPData/MLP_CW3/input/test/risky/S001C001P001R002A051_Kicking_other_person_risky.avi')

# /Users/rijul/Library/CloudStorage/OneDrive-UniversityofEdinburgh/MLPData/MLP_CW3/input/test/risky/S001C001P001R002A051_Kicking_other_person_risky.avi

if not cap.isOpened():
    print('Error while trying to read video. Please check path again')

frame_count = 0
clips = []

num_clips=0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    image = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(image=frame)['image']
    clips.append(frame)


    if len(clips) == 16:
        with torch.no_grad():
            num_clips += 1
            # if num_clips%15!=0:
            #     clips=[]
            #     continue

            input_frames = np.array(clips)
            input_frames = np.expand_dims(input_frames, axis=0)
            input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))
            input_frames = torch.tensor(input_frames, dtype=torch.float32)
            input_frames = input_frames.to(device)

            output = model(input_frames)

            # Process the attention maps
            attn_maps = attn_maps.squeeze(1)
            num_windows = attn_maps.shape[1]
            attn_maps = attn_maps.view(-1, num_windows, attn_maps.shape[-2], attn_maps.shape[-1])
            attn_maps = attn_maps.permute(0, 2, 3, 1)
            attn_maps = attn_maps.cpu().numpy()

            # Visualize the attention maps
            num_heads = attn_maps.shape[0]
            print("Shape of attention maps: ", attn_maps.shape)

            output_reshaped = attn_maps.reshape(90*90, 96, 8)

            plt.figure(figsize=(12, 6))
            plt.imshow(output_reshaped.mean(axis=0), cmap='viridis', aspect='auto')
            plt.title(f'Multi-Head Attention Visualization (Clip {frame_count // 16 + 1})')
            # plt.xlabel('Heads')
            # Remove the x-label
            plt.gca().xaxis.set_label_text('')
            plt.gca().xaxis.set_ticks([])
            plt.ylabel('Positions')
            plt.colorbar()
            plt.savefig(f'good_vid_attn_map_clip_{frame_count // 16 + 1}.png')
            plt.close()
            cv2.imwrite(f'good_vid_last_frame_clip_{frame_count // 16 + 1}.png', image)
            

        clips = []

    frame_count += 1

print("Number of clips: ",num_clips)

# Remove the hook
hook.remove()

cap.release()