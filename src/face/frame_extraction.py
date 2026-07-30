import cv2
import numpy as np

IMG_SIZE = 224

def extract_frames(video_path, target_fps=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        cap.release()
        raise ValueError("Invalid FPS.")

    frame_interval = max(int(round(original_fps / target_fps)), 1)
    frames, timestamps = [], []
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_idx % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            timestamps.append(frame_idx / original_fps)
        frame_idx += 1

    cap.release()
    return frames, timestamps


def detect_faces(frames, timestamps, detector):
    face_data = []
    for frame, timestamp in zip(frames, timestamps):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        faces = detector.get(frame_bgr)
        if len(faces) > 0:
            best_face = max(faces, key=lambda x: x.det_score)
            confidence = float(best_face.det_score)
            x1, y1, x2, y2 = best_face.bbox.astype(int)
            face_crop = frame[max(0, y1):y2, max(0, x1):x2]
            if face_crop.size != 0:
                # Resize + cast here so workers hand back small, ready-to-embed arrays
                face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE)).astype(np.uint8)
                face_data.append((face_resized, timestamp, confidence))
    return face_data