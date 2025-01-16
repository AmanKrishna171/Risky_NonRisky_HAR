from moviepy.editor import concatenate_videoclips, VideoFileClip
import os

# Directory containing the videos
video_dir = '/Users/rijul/Library/CloudStorage/OneDrive-UniversityofEdinburgh/MLPData/MLP_CW3/20230717_Training_a_Video_Classification_Model_from_Torchvision/src/merge_test_videos'

# Get a list of all the video files
video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]

# Create a VideoFileClip object for each video file
common_fps = 15  # Set this to your desired fps
video_clips = [VideoFileClip(os.path.join(video_dir, f)).set_duration(common_fps) for f in video_files]
# Concatenate all the video clips into a single video
final_clip = concatenate_videoclips(video_clips)

# Write the final video to a file
final_clip.write_videofile("/Users/rijul/Library/CloudStorage/OneDrive-UniversityofEdinburgh/MLPData/MLP_CW3/20230717_Training_a_Video_Classification_Model_from_Torchvision/src/merged_video.mp4")