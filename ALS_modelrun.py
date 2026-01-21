# ==============================
# ALS AUDIO → PREDICTION PIPELINE
# ==============================

# ---------- Imports ----------
import os
import numpy as np
import pandas as pd
import joblib
import librosa
import opensmile
from pydub import AudioSegment

# ---------- Load trained artifacts ----------
model = joblib.load("als_xgb_full_model.pkl")
scaler = joblib.load("als_scaler.pkl")
feature_names = joblib.load("als_feature_names.pkl")

# ---------- Initialize OpenSMILE ----------
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# ---------- Utility: Convert AAC → WAV ----------
def convert_aac_to_wav(audio_path, out_path="converted.wav"):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(out_path, format="wav")
    return out_path

# ---------- Utility: Check audio validity ----------
def is_audio_valid(wav_path, min_duration=1.0):
    y, sr = librosa.load(wav_path, sr=16000)
    duration = len(y) / sr
    energy = np.mean(np.abs(y))
    return duration >= min_duration and energy > 0.001

# ---------- Extract OpenSMILE features safely ----------
def extract_opensmile_features_safe(wav_path):
    if not is_audio_valid(wav_path):
        raise ValueError("Audio too short or silent for ALS analysis")

    df = smile.process_file(wav_path)

    if df is None or df.shape[0] == 0:
        raise ValueError("No voiced segments detected in audio")

    return df

# ---------- Prepare model input ----------
def prepare_als_input_from_audio(wav_path):
    df = extract_opensmile_features_safe(wav_path)

    # Create exactly ONE sample
    X_new = pd.DataFrame(0.0, index=[0], columns=feature_names)

    for col in feature_names:
        if col in df.columns:
            X_new.at[0, col] = df[col].mean()

    X_scaled = scaler.transform(X_new)
    return X_scaled

# ---------- MAIN PREDICTION FUNCTION ----------
def predict_als_from_audio(audio_path):
    try:
        # Convert if needed
        if audio_path.lower().endswith(".aac"):
            wav_path = convert_aac_to_wav(audio_path)
        else:
            wav_path = audio_path

        # Prepare input
        X_new = prepare_als_input_from_audio(wav_path)

        # Predict
        pred = model.predict(X_new)[0]
        prob = model.predict_proba(X_new)[0, 1]

        return {
            "prediction": "ALS" if pred == 1 else "Healthy",
            "confidence": float(prob if pred == 1 else 1 - prob)
        }

    except Exception as e:
        return {
            "error": str(e)
        }

 result = predict_als_from_audio("/content/input_audio.aac")
 print(result)
