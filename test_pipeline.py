import time
from src.inference.predict import predict_video
import json
import os

video_paths = [
    "data/testVideo1.mp4",
    "data/testVideo2.mp4",
    "data/testVideo3.mp4"
]

all_results = {}

for video_path in video_paths:
    print(f"Processing: {video_path}")

    result = predict_video(video_path)
    time.sleep(10)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    all_results[video_name] = result

    with open("all_outputs.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print("All outputs saved to all_outputs.json")
    
