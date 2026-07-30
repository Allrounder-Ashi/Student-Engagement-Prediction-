import numpy as np
import joblib
import tensorflow as tf
import json
from pathlib import Path
from ..face.preprocess import preprocess_video
from ..llm.generate_report import generate_learning_report
from .custom_layers import AttentionPooling, MaskedGlobalPooling

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
SCALER_PATH = MODEL_DIR / "feature_scaler.joblib"
CONFIG_PATH = BASE_DIR / "config.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

LABELS = config["labels"]                
CLASS_NAMES = config["class_names"]        
BEST_THRESHOLDS = config["best_thresholds"]  
SEQ_LEN = config["seq_len"]                
EMBED_DIM = config["embed_dim"]            
CUSTOM_OBJECTS = {
    "AttentionPooling": AttentionPooling,
    "MaskedGlobalPooling": MaskedGlobalPooling,
}
MODEL_PATH = MODEL_DIR / "final_model.keras"

_models = [
    tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects=CUSTOM_OBJECTS,
        compile=False
    )
]

scaler = joblib.load(SCALER_PATH)

def _get_padding_mask(features):
    """A frame is 'padding' if it's exactly all-zero (matches how the
    training notebook detects padded frames)."""
    return np.all(features == 0.0, axis=-1)


def _scale_features(features, mask):
    flat = features.reshape(-1, features.shape[-1])
    flat_scaled = scaler.transform(flat)
    flat_scaled[mask.reshape(-1)] = 0.0
    return flat_scaled.reshape(features.shape).astype(np.float32)


def _add_delta_features(scaled, mask):
    """Frame-to-frame delta ('motion') features, concatenated onto the
    raw scaled embeddings -- doubles the feature dim (matches Cell 6)."""
    real = ~mask
    diffs = np.zeros_like(scaled)
    diffs[1:, :] = scaled[1:, :] - scaled[:-1, :]
    both_real = real[1:] & real[:-1]
    diffs[1:, :] *= both_real[:, np.newaxis]
    return np.concatenate([scaled, diffs], axis=-1).astype(np.float32)


def build_model_input(raw_embeddings):
    """
    raw_embeddings already contains:
    [CLIP + delta] = 2048 dims
    """
    mask = _get_padding_mask(raw_embeddings)
    scaled = _scale_features(raw_embeddings, mask)
    return scaled.reshape(1, SEQ_LEN, -1)

def predict(features):
    """
    features: raw np.ndarray of shape (SEQ_LEN, EMBED_DIM), as returned by
    preprocess_video -- NOT yet scaled or delta-augmented.
    Returns: dict of {label_key: averaged sigmoid probability} across all
    N_SEEDS models, e.g. {"engagement_out": 0.73, "boredom_out": 0.21, ...}
    """
    model_input = build_model_input(features)
    per_seed_probs = {f"{label.lower()}_out": [] for label in LABELS}
    for model in _models:
        preds = model.predict(model_input, verbose=0)
        for label, p in zip(LABELS, preds):
            key = f"{label.lower()}_out"
            per_seed_probs[key].append(float(p.reshape(-1)[0]))
    return {key: float(np.mean(vals)) for key, vals in per_seed_probs.items()}


def predict_from_features(features):
    """
    Predict engagement, boredom, confusion and frustration (Low/High)
    from a raw (SEQ_LEN, EMBED_DIM) CLIP feature sequence.
    """
    probs = predict(features)
    results = decode_predictions(probs)
    return results


def decode_predictions(probs):
    """
    probs: dict of {label_key: averaged sigmoid probability}
    Applies the per-label tuned thresholds (NOT a flat 0.5) to decide
    Low vs High, matching how the training notebook evaluated the model.
    """
    results = {}
    for label in LABELS:
        key = f"{label.lower()}_out"
        threshold = BEST_THRESHOLDS[key]
        is_high = probs[key] > threshold
        results[label] = CLASS_NAMES[1] if is_high else CLASS_NAMES[0]
    return results


def predict_video(video_path, topic=None):
    """
    Complete inference pipeline from video to predictions + LLM report.
    """
    features = preprocess_video(video_path)
    if features is None:
        return None
    predictions = predict_from_features(features)
    print("Predictions:", predictions)
    if topic is None:
        topic = input("\nEnter the topic you want the AI Tutor to explain: ").strip()
        while topic == "":
            topic = input("Topic cannot be empty. Please enter a topic: ").strip()
    report = generate_learning_report(
        predictions["Engagement"],
        predictions["Boredom"],
        predictions["Confusion"],
        predictions["Frustration"],
        topic
    )
    return {
        "predictions": predictions,
        "topic": topic,
        "report": report,
    }
