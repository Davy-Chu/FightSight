import cv2
import os
import subprocess

def get_video_name(path):
    return os.path.splitext(os.path.basename(path))[0]

def extract_frames(video_path, output_dir, fps=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if int(count % round(frame_rate / fps)) == 0:
            filename = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            filename = filename.replace('\\', '/')
            print(filename, "saved")
            cv2.imwrite(filename, frame)
            saved += 1
        count += 1

    cap.release()

def extract_audio(video_path, output_path):
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",  # no video
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        output_path
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Audio extraction failed for: {video_path}")