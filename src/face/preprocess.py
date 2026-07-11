import numpy as np
from insightface.app import FaceAnalysis
from .frame_extraction import extract_frames, detect_faces
from .feature_extraction import (
    extract_embeddings,
    select_frames,
    create_sequence,
)

detector = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection"]
)
detector.prepare(ctx_id=1, det_size=(320, 320))

def preprocess_video(video_path):
    frames, timestamps = extract_frames(video_path)
    face_data = detect_faces(frames, timestamps, detector)
    if not face_data:
        return None

    faces = np.stack([f[0] for f in face_data])
    times = np.array([f[1] for f in face_data], dtype=np.float32)
    confs = np.array([f[2] for f in face_data], dtype=np.float32)

    embeddings = extract_embeddings(faces)
    selected = select_frames(embeddings, times, confs)
    sequence = create_sequence(embeddings, selected)
    return sequence