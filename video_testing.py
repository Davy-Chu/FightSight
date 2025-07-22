import os
import cv2
from utils.video.video_utils import extract_frames, get_video_name
from utils.video.pose_utils import detect_fall_intervals
from utils.video.event_summarizer import summarize_event_for_llm
from utils.video.fall_classifier import classify_fall_event_with_llm
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
VIDEO_PATH = "data/raw_data/fight1.mp4"
FRAMES_DIR = f"data/frames/{get_video_name(VIDEO_PATH)}"
FPS = 5  # Must match your extract_frames FPS
print(VIDEO_PATH, FRAMES_DIR)
# Step 1: Extract frames
extract_frames(VIDEO_PATH, FRAMES_DIR, fps=FPS)

# Step 2: Load frames into memory
frame_files = sorted(os.listdir(FRAMES_DIR))
frames = [cv2.imread(os.path.join(FRAMES_DIR, f)) for f in frame_files]

# Step 3: Detect fall intervals
fall_intervals = detect_fall_intervals(frames, fps=FPS)
print(f"[DEBUG] Detected {len(fall_intervals)} fall intervals: {fall_intervals}")
# Step 4: Summarize and classify
# for start, end in fall_intervals:
#     event = {
#         "start_time": start,
#         "end_time": end,
#         "duration": end - start,
#         "fall_type": "fall"
#     }
#     summary = summarize_event_for_llm(event)
#     print("Summary: ", summary)
#     classification = classify_fall_event_with_llm(summary)
#     print(f"[EVENT] {classification.upper()} at {start:.2f}s (duration: {end - start:.2f}s)")