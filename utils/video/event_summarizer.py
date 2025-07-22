def summarize_event_for_llm(event):
    """
    Convert structured grounded event data into a textual description for LLM classification.
    
    event: dict with keys:
        - 'start_time': float
        - 'end_time': float
        - 'duration': float
        - 'fall_type': str (initial guess: 'fall')
        - 'initial_fall_frame': int
        - 'fall_pose_info': dict (optional, metadata about pose)
        - 'context': list of dicts (optional frames before/after with pose states)
    """
    start = round(event["start_time"], 2)
    end = round(event["end_time"], 2)
    duration = round(event["duration"], 2)

    summary = f"The fighter falls at {start}s and stays grounded until {end}s for {duration} seconds."

    # Optional pose info
    if "fall_pose_info" in event:
        pose_info = event["fall_pose_info"]
        details = []
        if "fall_velocity" in pose_info:
            vel = pose_info["fall_velocity"]
            details.append(f"fall velocity is estimated at {vel:.2f}")
        if "impact_location" in pose_info:
            details.append(f"impact seems to occur at {pose_info['impact_location']}")
        if details:
            summary += " " + ", ".join(details) + "."

    # Optional contextual info
    if "context" in event:
        num_frames = len(event["context"])
        strikes = [f for f in event["context"] if f.get("event") == "strike"]
        if strikes:
            summary += f" The fall is preceded by {len(strikes)} strike(s) within the prior {num_frames} frames."

    return summary