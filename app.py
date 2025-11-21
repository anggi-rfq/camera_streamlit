import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

CANDIDATE_PATHS = [
    "/mnt/data/camera_dataset.csv",
    "/content/drive/MyDrive/camera_dataset.csv",
    "camera_dataset.csv"
]
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "models_camera.joblib")

st.set_page_config(page_title="Camera Premium Predictor", layout="centered")
st.title("Camera Premium Predictor")
st.write("Aplikasi sederhana: prediksi apakah sebuah kamera termasuk **premium** berdasarkan spesifikasi.\
         Jika model belum ada, aplikasi akan melatih model secara otomatis (menggunakan dataset yang tersedia).")

def find_dataset():
    for p in CANDIDATE_PATHS:
        if os.path.exists(p):
            return p
    return None

def load_and_prepare(path):
    df = pd.read_csv(path)
    if 'Price' not in df.columns:
        st.error("Kolom 'Price' tidak ditemukan pada dataset. Tidak bisa melanjutkan.")
        st.stop()
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    q3 = df['Price'].quantile(0.75)
    df['is_premium'] = (df['Price'] >= q3).astype(int)

    X = df.select_dtypes(include=[np.number]).drop(columns=['is_premium'], errors='ignore')
    if 'Price' in X.columns:
        X = X.drop(columns=['Price'])
    X = X.fillna(X.median())
    y = df['is_premium']
    return df, X, y

def train_bundle(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    svc = SVC(kernel='rbf', C=3, probability=True, random_state=42)
    voting = VotingClassifier([('rf', rf), ('svc', svc)], voting='soft')
    voting.fit(X_train, y_train)
    # eval
    y_pred = voting.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    # bundle
    bundle = {"model": voting, "scaler": scaler, "feature_names": list(X.columns), "eval": {"acc": acc, "report": rep, "cm": cm}}
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)
    return bundle

dataset_path = find_dataset()
if dataset_path is None:
    st.warning("Dataset tidak ditemukan. Untuk menjalankan app ini Anda perlu upload file 'camera_dataset.csv' ke repo atau drive.")
    st.info("Cek path yang dicari: " + ", ".join(CANDIDATE_PATHS))
else:
    st.write("Dataset ditemukan di:", dataset_path)

if os.path.exists(MODEL_FILE):
    st.success("Memuat model dari file.")
    bundle = joblib.load(MODEL_FILE)
    model = bundle.get("model")
    scaler = bundle.get("scaler")
    feature_names = bundle.get("feature_names")
else:
    st.info("Model belum ditemukan. Menyiapkan pelatihan (akan memakan waktu beberapa saat).")
    if dataset_path is None:
        st.stop()
    df, X, y = load_and_prepare(dataset_path)
    bundle = train_bundle(X, y)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    st.success("Pelatihan selesai dan model disimpan.")

if "eval" in bundle:
    eval_info = bundle["eval"]
    st.subheader("Evaluasi model (test set)")
    st.write("Accuracy:", eval_info["acc"])
    st.text("Classification report:\n" + str(eval_info["report"]))
    st.text("Confusion matrix:\n" + str(eval_info["cm"]))

st.subheader("Prediksi satu kamera (isi nilai numerik fitur)")

st.write("Masukkan nilai numerik untuk setiap fitur di bawah (urut sama seperti feature names). Jika tidak tahu, isi 0.")
st.write("Urutan fitur (feature_names):")
st.write(feature_names)

inputs = []
cols = st.columns(2)
for i, fname in enumerate(feature_names):
    c = cols[i % 2]
    val = c.text_input(fname, value="0")
    inputs.append(val)

if st.button("Predict"):
    try:
        vals = [float(x) for x in inputs]
    except:
        st.error("Semua input harus angka (contoh: 12.0).")
        st.stop()
    arr = np.array(vals).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)[0]
    prob = model.predict_proba(arr_scaled)[0,1] if hasattr(model, "predict_proba") else None
    if pred == 1:
        st.success("Prediksi: PREMIUM")
    else:
        st.info("Prediksi: NON-PREMIUM")
    if prob is not None:
        st.write(f"Probabilitas premium: {prob:.3f}")

st.markdown("---")
st.write("Catatan: app ini memakai fitur numerik yang tersedia di dataset. Jika dataset punya banyak kolom teks/format berbeda, lakukan preprocessing (parsing) terlebih dahulu sebelum melatih model.")
