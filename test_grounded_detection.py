import os
import cv2
import numpy as np
from utils.video.pose_utils import get_multiple_pose_keypoints, is_person_lying_down_adaptive, is_person_lying_down, is_fighter_grounded_strict, detect_camera_angle
import mediapipe as mp

def analyze_frame(frame_path, frame_num, time):
    """Analyze a specific frame for grounded detection"""
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Could not load frame {frame_path}")
        return
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)
    
    keypoints_list = get_multiple_pose_keypoints(frame, pose)
    
    print(f"\n=== Frame {frame_num} (Time: {time:.1f}s) ===")
    print(f"Number of pose detections: {len(keypoints_list)}")
    
    for i, keypoints in enumerate(keypoints_list):
        if keypoints is None:
            continue
            
        # Get key points for analysis
        l_hip = keypoints[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
        r_hip = keypoints[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
        l_shoulder = keypoints[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = keypoints[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_knee = keypoints[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = keypoints[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]
        
        print(f"\nPose {i}:")
        print(f"  Left hip: ({l_hip[0]:.3f}, {l_hip[1]:.3f}, vis: {l_hip[2]:.3f})")
        print(f"  Right hip: ({r_hip[0]:.3f}, {r_hip[1]:.3f}, vis: {r_hip[2]:.3f})")
        print(f"  Left shoulder: ({l_shoulder[0]:.3f}, {l_shoulder[1]:.3f}, vis: {l_shoulder[2]:.3f})")
        print(f"  Right shoulder: ({r_shoulder[0]:.3f}, {r_shoulder[1]:.3f}, vis: {r_shoulder[2]:.3f})")
        print(f"  Left knee: ({l_knee[0]:.3f}, {l_knee[1]:.3f}, vis: {l_knee[2]:.3f})")
        print(f"  Right knee: ({r_knee[0]:.3f}, {r_knee[1]:.3f}, vis: {r_knee[2]:.3f})")
        
        # Calculate metrics
        hips_y = np.mean([l_hip[1], r_hip[1]])
        shoulders_y = np.mean([l_shoulder[1], r_shoulder[1]])
        knees_y = np.mean([l_knee[1], r_knee[1]]) if l_knee[2] > 0.5 and r_knee[2] > 0.5 else None
        
        shoulder_hip_diff = abs(shoulders_y - hips_y)
        lowest_point = max(hips_y, shoulders_y)
        if knees_y is not None:
            lowest_point = max(lowest_point, knees_y)
            hip_knee_diff = abs(hips_y - knees_y)
            print(f"  Hip-knee difference: {hip_knee_diff:.3f}")
        
        print(f"  Shoulder-hip difference: {shoulder_hip_diff:.3f}")
        print(f"  Lowest point: {lowest_point:.3f}")
        
        # Test detection methods
        camera_angle = detect_camera_angle(None, keypoints)
        print(f"  Camera angle: {camera_angle}")
        
        lying_down_adaptive = is_person_lying_down_adaptive(keypoints, 0.5)
        lying_down_original = is_person_lying_down(keypoints, 0.5)
        grounded_strict = is_fighter_grounded_strict(keypoints, 0.5)
        
        print(f"  Lying down (adaptive): {lying_down_adaptive}")
        print(f"  Lying down (original): {lying_down_original}")
        print(f"  Grounded (strict): {grounded_strict}")
    
    pose.close()

# Analyze frames around the fall at 64 seconds
frames_dir = "data/frames/fight1"
fps = 5

# Analyze frames from 63s to 66s (around the fall)
for frame_num in range(315, 330):  # frames 315-329 correspond to 63.0s to 65.8s
    time = frame_num / fps
    frame_path = os.path.join(frames_dir, f"frame_{frame_num:04d}.jpg")
    
    if os.path.exists(frame_path):
        analyze_frame(frame_path, frame_num, time)
    else:
        print(f"Frame {frame_num} not found") 