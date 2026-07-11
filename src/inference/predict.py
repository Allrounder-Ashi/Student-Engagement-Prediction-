import numpy as np
import joblib
import tensorflow as tf
import json
from ..face.preprocess import preprocess_video
from ..llm.generate_report import generate_learning_report


from .custom_layers import AttentionPooling





MODEL_PATH = "models/best_model.keras"
SCALER_PATH = "models/feature_scaler.joblib"
CONFIG_PATH = "config.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

LABELS = config["labels"]
CLASS_NAMES = config["class_names_per_label"]    

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "AttentionPooling": AttentionPooling
    },
    compile=False
)

scaler = joblib.load(SCALER_PATH)


def predict(features):
    """
    features: numpy array of shape (16, 2048)
    """

    # Add batch dimension
    features = features.reshape(1, 16, 2048)

    # Predict
    predictions = model.predict(features, verbose=0)

    return predictions


def predict_from_features(features):
    """
    Predict engagement, boredom, confusion and frustration
    from extracted feature vectors.
    """
    predictions = predict(features)
    results = decode_predictions(predictions)
    return results

def decode_predictions(predictions):
    engagement = np.argmax(predictions[0], axis=1)[0]
    boredom = np.argmax(predictions[1], axis=1)[0]
    confusion = np.argmax(predictions[2], axis=1)[0]
    frustration = np.argmax(predictions[3], axis=1)[0]
    return {
        "Engagement": str(CLASS_NAMES["Engagement"][engagement]),
        "Boredom": str(CLASS_NAMES["Boredom"][boredom]),
        "Confusion": str(CLASS_NAMES["Confusion"][confusion]),
        "Frustration": str(CLASS_NAMES["Frustration"][frustration]),
    }

def predict_video(video_path):
    """
    Complete inference pipeline from video to predictions.
    """
    features = preprocess_video(video_path)

    if features is None:
        return None

    predictions = predict_from_features(features)

    print("Predictions:", predictions)

    report = generate_learning_report(
        engagement=predictions["Engagement"],
        boredom=predictions["Boredom"],
        confusion=predictions["Confusion"],
        frustration=predictions["Frustration"],
        topic="Your Learning Topic"
    )

    return {
        "predictions": predictions,
        "report": report
    }