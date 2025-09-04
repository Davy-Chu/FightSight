# FightSight
🥊 FightSight — your AI cornerman

FightSight is an AI-powered toolkit that auto-annotates MMA fights. Drop in a bout video and get back timestamped events (knockdowns, takedowns, slips), using a pipeline of computer vision, audio transcription, and an LLM judge.

License: MIT • Stack: Python · OpenCV · MediaPipe · FFmpeg · Whisper · OpenAI (LLM)
what it does

Extracts frames & audio from your fight video

Uses pose estimation + smart heuristics to detect falls/grounded intervals

Camera-cut guard (SSIM + buffers) to avoid false “teleport” falls

Transcribes commentary with Whisper (optional, boosts context)

Summarizes each event and asks an LLM to label it: knockdown • takedown • slip

Caches LLM results so repeat runs are fast & free

Exports clean JSON with event type, start/end, and duration

how it works

Chat with your cornerman (okay, the CLI) like this:

“analyze fight1.mp4” → extract frames & audio

“find falls” → detect grounded intervals with pose

“classify events” → LLM labels each interval

“export json” → write annotations to file

Under the hood:

video → frames (OpenCV) + audio (FFmpeg)

pose → MediaPipe keypoints → grounded detection (hips/knees/head + duration)

stability → SSIM camera-cut detection, standing-buffer, teleport-guard

audio → Whisper transcript (optional)

LLM → natural-language event summary → classification

output → JSON (and optional on-video overlays)

get started

Prereqs: Python 3.10+, FFmpeg on PATH
Verify: ffmpeg -version

# 1) clone
git clone https://github.com/yourname/FightSight.git
cd FightSight

# 2) create env & install
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt

# 3) add your OpenAI key
copy .env.example .env   # (Windows)  or: cp .env.example .env
# then edit .env and set: OPENAI_API_KEY=sk-...


Put a video at data/raw_data/fight1.mp4.

# 4) extract frames & audio (choose fps; 5 is a good start)
python extract_frame.py --video data/raw_data/fight1.mp4 --fps 5

# 5) run end-to-end test (detect & classify events)
python video_testing.py


You’ll see something like:

[INFO] Detected 2 fall intervals
Summary: The fighter falls at 65.00s and stays grounded until 67.20s for 2.20s.
[EVENT] TAKEDOWN at 65.00s (duration: 2.20s)
[EVENT] SLIP at 120.00s (duration: 1.00s)

repo layout
FightSight/
├─ data/
│  ├─ raw_data/            # your input videos
│  ├─ frames/<video>/      # extracted frames
│  ├─ audio/               # extracted wav
│  ├─ cache/               # LLM result cache (JSON)
│  └─ transcripts/         # optional Whisper saves
├─ utils/
│  ├─ video/
│  │  ├─ video_utils.py        # frame/audio extraction
│  │  ├─ pose_utils.py         # pose, grounded detection, cut guards
│  │  └─ event_summarizer.py   # turns events → LLM-ready summaries
│  └─ llm/
│     └─ fall_classifier.py    # OpenAI call + prompt hashing cache
├─ extract_frame.py            # CLI: extract frames + audio
├─ video_testing.py            # end-to-end demo
├─ requirements.txt
└─ .env.example

tuning tips

No events? Increase --fps to 5–10; lower STAND_BUFFER in pose_utils.py.

False falls at cuts? Lower SSIM threshold (more sensitive), increase skip buffer after cuts, ensure we flush pending grounded frames on cut.

Everything “slip”? Ensure summaries include duration; longer grounded intervals skew toward takedown/knockdown.

Config knobs (in pose_utils.py):
SSIM_THRESH · STAND_BUFFER · MAX_DROP · skip_next = int(fps*0.5) · min_duration

why fightsight exists

Scrubbing for highlights eats time. FightSight gives coaches, analysts, and fans an instant timeline of pivotal moments—ground truth first, LLM polish second—so you can focus on insight, not rewinding.

built with 💙 for corners, coaches & fight nerds

PRs welcome. Issues too.
