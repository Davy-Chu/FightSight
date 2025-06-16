import whisper
import os

def transcribe_audio_whisper(audio_path, model_size="medium"):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"[INFO] Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size, device="cpu")
    result = model.transcribe(audio_path, fp16=False, word_timestamps=False)

    print(f"[INFO] Transcribing: {audio_path}")
    # result = model.transcribe(audio_path)
    
    print(f"[INFO] Transcription complete.")
    return result["segments"]