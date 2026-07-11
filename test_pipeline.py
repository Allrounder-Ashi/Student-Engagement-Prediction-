from src.inference.predict import predict_video
video_path = "data/test.mp4"   # Apni video ka path
result = predict_video(video_path)
print(result)