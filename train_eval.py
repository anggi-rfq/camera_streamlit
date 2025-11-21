# train_eval.py
"""
Script sederhana untuk:
- load dataset camera
- membuat label is_premium (Price >= Q3)
- preprocessing sederhana (numeric only, fillna median, scaling)
- melatih ensemble (RandomForest + SVC) dengan VotingClassifier (soft)
- evaluasi pada test set, cetak report dan confusion matrix
- simpan model + scaler + fitur ke models/models_camera.joblib

Cocok dijalankan di: lokal, Google Colab, atau CI sebelum deploy.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ---------- Config: path dataset (cek beberapa lokasi umum)
CANDIDATE_PATHS = [
    "/mnt/data/camera_dataset.csv",                     # environment lokal tertentu
    "/content/drive/MyDrive/camera_dataset.csv",        # Google Drive (Colab)
    "camera_dataset.csv"                                # repo root (deploy)
]
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "models_camera.joblib")

def find_dataset():
    for p in CANDIDATE_PATHS:
        if os.path.exists(p):
            return p
    return None

def load_dataset(path):
    print("Loading dataset from:", path)
    df = pd.read_csv(path)
    return df

def prepare_data(df):
    # pastikan kolom Price ada, konversi numeric
    if 'Price' not in df.columns:
        raise ValueError("Kolom 'Price' tidak ditemukan di dataset. Pastikan ada kolom harga.")
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # buat label premium (1 jika Price >= Q3)
    q3 = df['Price'].quantile(0.75)
    df['is_premium'] = (df['Price'] >= q3).astype(int)
    print(f"Q3 price (threshold premium): {q3}")

    # pilih fitur numerik (kecuali target)
    X = df.select_dtypes(include=[np.number]).drop(columns=['is_premium'], errors='ignore')
    # hilangkan kolom Price jika tidak ingin kebocoran (opsional)
    # Jika Anda mau gunakan Price sebagai fitur, hapus komentar baris berikut.
    if 'Price' in X.columns:
        X = X.drop(columns=['Price'])

    # isi missing numeric dengan median
    X = X.fillna(X.median())

    y = df['is_premium']
    return X, y

def train_and_evaluate(X, y):
    # scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # models
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    svc = SVC(kernel='rbf', C=3, probability=True, random_state=42)
    voting = VotingClassifier([('rf', rf), ('svc', svc)], voting='soft')

    # train
    print("Training ensemble (RandomForest + SVC)...")
    voting.fit(X_train, y_train)

    # eval
    y_pred = voting.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy on test set: {acc:.4f}")
    print("\nClassification report:\n", report)
    print("Confusion matrix:\n", cm)

    # simpan bundle
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        "model": voting,
        "scaler": scaler,
        "feature_names": list(X.columns)
    }
    joblib.dump(bundle, MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")

    return voting, scaler, X.columns, acc, report, cm

def main():
    path = find_dataset()
    if path is None:
        print("Dataset tidak ditemukan. Silakan letakkan file 'camera_dataset.csv' di salah satu lokasi berikut:")
        for p in CANDIDATE_PATHS:
            print(" -", p)
        return

    df = load_dataset(path)
    X, y = prepare_data(df)
    train_and_evaluate(X, y)

if __name__ == "__main__":
    main()
