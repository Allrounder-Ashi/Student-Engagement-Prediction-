import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

FPS = 5
IMG_SIZE = 224

NUM_SELECTED_FRAMES = 16
NUM_CHUNKS = 8
NUM_TRANSITION_FRAMES = 8
MEDOID_TIE_EPSILON = 1e-3

# Backbone for extracting face embeddings (global, reused)
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))

def select_frames(embeddings, timestamps, confidences,
                   num_chunks=NUM_CHUNKS, num_transition_frames=NUM_TRANSITION_FRAMES,
                   tie_epsilon=MEDOID_TIE_EPSILON):
    total_frames = len(embeddings)
    selected_indices = set()

    chunk_size = total_frames / num_chunks
    for i in range(num_chunks):
        start, end = int(i * chunk_size), int((i + 1) * chunk_size)
        chunk_indices = list(range(start, end))
        if len(chunk_indices) == 0:
            continue
        chunk_embeddings = embeddings[chunk_indices]
        chunk_confidences = confidences[chunk_indices]

        similarity = cosine_similarity(chunk_embeddings)
        avg_similarity = similarity.mean(axis=1)

        best_score = avg_similarity.max()
        # Any local index within epsilon of the best score is a tied candidate
        tied_candidates = [
            local_idx for local_idx, score in enumerate(avg_similarity)
            if best_score - score <= tie_epsilon
        ]

        if len(tied_candidates) == 1:
            medoid_local = tied_candidates[0]
        else:
            # Tie-break: pick the candidate with the highest RetinaFace detection confidence
            medoid_local = max(tied_candidates, key=lambda local_idx, conf=chunk_confidences: conf[local_idx])

        medoid_global = chunk_indices[medoid_local]
        selected_indices.add(medoid_global)

    # Transition frames (biggest confidence-weighted jumps)
    transition_scores = []
    for i in range(total_frames - 1):
        distance = cosine_distances(embeddings[i:i+1], embeddings[i+1:i+2])[0][0]
        score = distance * min(confidences[i], confidences[i+1])
        transition_scores.append((score, i))
    transition_scores.sort(reverse=True)
    for _, idx in transition_scores[:num_transition_frames]:
        selected_indices.add(idx)

    return sorted(selected_indices, key=lambda x: timestamps[x])


def create_sequence(embeddings, selected_indices, sequence_length=NUM_SELECTED_FRAMES):
    selected_embeddings = embeddings[selected_indices]
    if len(selected_embeddings) > sequence_length:
        selected_embeddings = selected_embeddings[:sequence_length]
    elif len(selected_embeddings) < sequence_length:
        padding = np.zeros((sequence_length - len(selected_embeddings), embeddings.shape[1]))
        selected_embeddings = np.vstack([selected_embeddings, padding])
    return selected_embeddings

def extract_embeddings(faces):
    """faces: numpy array of shape (N, 224, 224, 3)
    returns: embeddings of shape (N, 2048)
    """

    faces = faces.astype(np.float32)
    faces = preprocess_input(faces)

    embeddings = base_model.predict(
        faces,
        batch_size=64,
        verbose=0
    )

    return embeddings