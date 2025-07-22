import argparse
from utils.video.video_utils import extract_frames, extract_audio, get_video_name
import os

def main(video_path):
    video_name = get_video_name(video_path)

    # Set output directories
    frames_dir = os.path.join("data/frames", video_name)
    audio_path = os.path.join("data/audio", f"{video_name}.wav")

    print(f"[INFO] Extracting frames to: {frames_dir}")
    extract_frames(video_path, frames_dir)

    print(f"[INFO] Extracting audio to: {audio_path}")
    extract_audio(video_path, audio_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames and audio from a video")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video")
    args = parser.parse_args()
    main(args.video)