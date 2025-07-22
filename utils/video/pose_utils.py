import cv2 as cv
import mediapipe as mp
import numpy as np

def get_pose_keypoints(frame, pose_model):
    results = pose_model.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None
    return [(lm.x, lm.y, lm.visibility) for lm in results.pose_landmarks.landmark]

def get_multiple_pose_keypoints(frame, pose_model):
    """Get pose keypoints for multiple people in the frame"""
    results = pose_model.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return []
    
    # For now, MediaPipe Pose only detects one person at a time
    # We'll need to use a different approach for multiple people
    # Let's try using MediaPipe Pose with different regions of the frame
    keypoints_list = []
    
    # Get the full frame pose
    full_keypoints = [(lm.x, lm.y, lm.visibility) for lm in results.pose_landmarks.landmark]
    if full_keypoints:
        keypoints_list.append(full_keypoints)
    
    # Try left half of frame
    h, w = frame.shape[:2]
    left_half = frame[:, :w//2]
    left_results = pose_model.process(cv.cvtColor(left_half, cv.COLOR_BGR2RGB))
    if left_results.pose_landmarks:
        left_keypoints = [(lm.x * 0.5, lm.y, lm.visibility) for lm in left_results.pose_landmarks.landmark]
        keypoints_list.append(left_keypoints)
    
    # Try right half of frame
    right_half = frame[:, w//2:]
    right_results = pose_model.process(cv.cvtColor(right_half, cv.COLOR_BGR2RGB))
    if right_results.pose_landmarks:
        right_keypoints = [(lm.x * 0.5 + 0.5, lm.y, lm.visibility) for lm in right_results.pose_landmarks.landmark]
        keypoints_list.append(right_keypoints)
    
    return keypoints_list

def detect_falling_motion_simple(prev_keypoints, curr_keypoints, align_threshold=0.12, drop_threshold=0.10):
    """Detect if hips and shoulders become aligned suddenly (fall event)."""
    if prev_keypoints is None or curr_keypoints is None:
        return False
    # Shoulders and hips y
    prev_hips_y = np.mean([
        prev_keypoints[mp.solutions.pose.PoseLandmark.LEFT_HIP.value][1],
        prev_keypoints[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value][1]
    ])
    prev_shoulders_y = np.mean([
        prev_keypoints[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value][1],
        prev_keypoints[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value][1]
    ])
    curr_hips_y = np.mean([
        curr_keypoints[mp.solutions.pose.PoseLandmark.LEFT_HIP.value][1],
        curr_keypoints[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value][1]
    ])
    curr_shoulders_y = np.mean([
        curr_keypoints[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value][1],
        curr_keypoints[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value][1]
    ])
    # Difference before and after
    prev_diff = abs(prev_hips_y - prev_shoulders_y)
    curr_diff = abs(curr_hips_y - curr_shoulders_y)
    # Detect a sudden drop to alignment
    return prev_diff > align_threshold and curr_diff < drop_threshold

def detect_falling_motion_gradual(prev_diffs, curr_diff, align_threshold=0.08, drop_threshold=0.15, window=3):
    """Detect if hips and shoulders become aligned gradually within a window of frames."""
    # prev_diffs: list of previous diffs (length up to window)
    # curr_diff: current diff
    # If any of the previous diffs is > align_threshold and current diff < drop_threshold, return True
    return any(d > align_threshold for d in prev_diffs) and curr_diff < drop_threshold

def keypoints_center(keypoints):
    """Compute the center (mean x, mean y) of visible keypoints."""
    coords = np.array([(kp[0], kp[1]) for kp in keypoints if kp[2] > 0.3])
    if len(coords) == 0:
        return np.array([0.0, 0.0])
    return np.mean(coords, axis=0)

def detect_fall_intervals(frames, fps, min_duration=0.3, max_duration=10.0):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)
    prev_keypoints_list = None
    window = int(2 * fps)  # 2 seconds window for gradual fall
    stand_window = int(2 * fps)  # 2 seconds required standing before new fall
    prev_diffs_list = []  # List of lists: for each person, keep last N diffs
    stand_counters = []  # For each person, count consecutive standing frames
    all_events = []
    current_group = []
    for i, frame in enumerate(frames):
        curr_keypoints_list = get_multiple_pose_keypoints(frame, pose)
        fall_detected = False
        fall_frame = None
        if prev_keypoints_list and curr_keypoints_list:
            # For each person in current frame, find closest in previous frame
            used_prev = set()
            new_prev_diffs_list = []
            new_stand_counters = []
            for curr_idx, curr_kp in enumerate(curr_keypoints_list):
                curr_center = keypoints_center(curr_kp)
                min_dist = float('inf')
                best_prev = None
                best_prev_idx = None
                for j, prev_kp in enumerate(prev_keypoints_list):
                    if j in used_prev:
                        continue
                    prev_center = keypoints_center(prev_kp)
                    dist = np.linalg.norm(curr_center - prev_center)
                    if dist < min_dist:
                        min_dist = dist
                        best_prev_idx = j
                        best_prev = prev_kp
                if best_prev is not None:
                    used_prev.add(best_prev_idx)
                    # Compute diffs for gradual detection
                    prev_diffs = prev_diffs_list[best_prev_idx] if len(prev_diffs_list) > best_prev_idx else []
                    stand_counter = stand_counters[best_prev_idx] if len(stand_counters) > best_prev_idx else stand_window
                    curr_hips_y = np.mean([
                        curr_kp[mp.solutions.pose.PoseLandmark.LEFT_HIP.value][1],
                        curr_kp[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value][1]
                    ])
                    curr_shoulders_y = np.mean([
                        curr_kp[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value][1],
                        curr_kp[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value][1]
                    ])
                    curr_diff = abs(curr_hips_y - curr_shoulders_y)
                    # Standing logic: if above align_threshold, increment counter; else reset
                    if curr_diff > 0.08:
                        stand_counter += 1
                        if stand_counter > stand_window:
                            stand_counter = stand_window
                    else:
                        stand_counter = 0
                    # Only allow fall detection if standing for at least stand_window frames
                    if stand_counter >= stand_window:
                        if detect_falling_motion_gradual(prev_diffs, curr_diff, window=window):
                            fall_detected = True
                            fall_frame = i
                            stand_counter = 0  # Reset after fall
                            break
                    # Update diffs and standing counter for this person
                    if len(prev_diffs) >= window:
                        prev_diffs = prev_diffs[1:]
                    prev_diffs = prev_diffs + [curr_diff]
                    new_prev_diffs_list.append(prev_diffs)
                    new_stand_counters.append(stand_counter)
                else:
                    new_prev_diffs_list.append([])
                    new_stand_counters.append(stand_window)
            prev_diffs_list = new_prev_diffs_list
            stand_counters = new_stand_counters
        else:
            # No previous keypoints, just initialize diffs and standing counters
            prev_diffs_list = [[] for _ in curr_keypoints_list] if curr_keypoints_list else []
            stand_counters = [stand_window for _ in curr_keypoints_list] if curr_keypoints_list else []
        if fall_detected:
            if not current_group:
                current_group = [fall_frame]
            else:
                # If a new fall is detected within the standing window after a previous fall, extend the interval
                if fall_frame - current_group[-1] <= stand_window:
                    current_group.append(fall_frame)
                else:
                    # Otherwise, finalize the previous group and start a new one
                    start_idx, end_idx = current_group[0], current_group[-1]
                    duration = (end_idx - start_idx + 1) / fps
                    if min_duration <= duration <= max_duration:
                        all_events.append((start_idx / fps, end_idx / fps))
                    current_group = [fall_frame]
        else:
            if current_group:
                start_idx, end_idx = current_group[0], current_group[-1]
                duration = (end_idx - start_idx + 1) / fps
                if min_duration <= duration <= max_duration:
                    all_events.append((start_idx / fps, end_idx / fps))
                current_group = []
        prev_keypoints_list = curr_keypoints_list
    pose.close()
    # Finalize last group if any
    if current_group:
        start_idx, end_idx = current_group[0], current_group[-1]
        duration = (end_idx - start_idx + 1) / fps
        if min_duration <= duration <= max_duration:
            all_events.append((start_idx / fps, end_idx / fps))
    return all_events
