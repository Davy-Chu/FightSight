from utils.video_utils import extract_frames

# Testing extract_frames
video_path = "data/raw_data/fight1.mp4"  
output_dir = "data/frames"
frame_interval = 30  # Save every 30th frame

# Convert frame_interval to fps (if video is 30fps, we want 1 frame per second)
extract_frames(video_path, output_dir, fps=30/frame_interval)
