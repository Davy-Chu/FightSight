from utils.video.video_utils import extract_frames
from utils.audio.audio_utils import extract_audio_from_video
from utils.audio.transcription_utils import transcribe_audio_whisper
from utils.audio.llm_utils import analyze_commentary_with_llm
import json
import os
# Testing extract_frames
video_path = "data/raw_data/fight1.mp4"  
output_dir = "data/frames"
frame_interval = 30  # Save every 30th frame
def format_segments(segments):
    return [
        {
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip()
        }
        for seg in segments
    ]
# Convert frame_interval to fps (if video is 30fps, we want 1 frame per second)
# extract_frames(video_path, output_dir, fps=30/frame_interval)

# Testing extract_audio_from_video
# audio_path = extract_audio_from_video(video_path)
# print("audio_path: ", audio_path)

# Testing transcribe_audio_whisper
text = transcribe_audio_whisper("C:/Users/badjo/OneDrive/Documents/FightSight/data/raw_data/fight2.mp4")
print("text: ")
for seg in format_segments(text):
    print(seg)


#Testing analyze_segments_with_llm
events_json = analyze_commentary_with_llm(text)

print("events_json: ")
print(events_json)



