import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('2025a7ps0603p_model.keras')

emotions = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def predict_emotion(file_path):

    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    
    y, _ = librosa.effects.trim(y, top_db=30)
    
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_spec = librosa.power_to_db(spec, ref=np.max)
    
    if log_spec.shape[1] < 128:
        log_spec = np.pad(log_spec, ((0,0), (0, 128 - log_spec.shape[1])), mode='constant')
    input_data = log_spec[:, :128].reshape(1, 128, 128, 1)

    prediction = model.predict(input_data, verbose=0)
    emotion_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    print("-" * 30)
    print(f"File: {file_path}")
    print(f"Predicted Emotion: {emotions[emotion_idx].upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    # I tested this with a file from the dataset. Please replace with your own file path.
    test_file = 'aud.wav'
    predict_emotion(test_file)