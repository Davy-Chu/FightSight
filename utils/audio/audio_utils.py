from moviepy.editor import VideoFileClip
import os

def extract_audio_from_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    path_name = video_path.split("/")[-1].split(".")[0];
    output_audio_path = "data/audio/" + path_name + ".wav"
    print("output_audio_path: ", output_audio_path)

    print(f"[INFO] Extracting audio from {video_path}...")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path)
    print(f"[INFO] Audio saved to {output_audio_path}")

    return output_audio_path