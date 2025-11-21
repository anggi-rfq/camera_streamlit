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

st.set_page_config(page_title="Camera Premium Predictor (Pilih Model)", layout="centered")
st.title("Camera Premium Predictor — Pilih Model")
st.write("Pilih model untuk prediksi: Voting (ensemble), RandomForest, atau SVC. \
Jika model belum ada, app akan melatih ketiganya (akan memakan waktu).")

def find_dataset():
    for p in CANDIDATE_PATHS:
        if os.path.exists(p):
            return p
    return None

def load_and_prepare(path):
    df = pd.read_csv(path)
    if 'Price' not in df.columns:
        st.error("Kolom 'Price' tidak ditemukan di dataset. Tidak bisa melanjutkan.")
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

def train_all(X, y, save_file=MODEL_FILE):
    st.info("Melatih RandomForest, SVC, dan Voting (ensemble). Tunggu sebentar...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    svc = SVC(kernel='rbf', C=3, probability=True, random_state=42)
    voting = VotingClassifier([('rf', rf), ('svc', svc)], voting='soft')

    rf.fit(X_train, y_train)
    svc.fit(X_train, y_train)
    voting.fit(X_train, y_train)

    results = {}
    for name, model in [('RandomForest', rf), ('SVC', svc), ('Voting', voting)]:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rep = classification_report(y_test, y_pred, output_dict=False)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {"accuracy": acc, "report": rep, "cm": cm}

    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        "rf": rf,
        "svc": svc,
        "voting": voting,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "eval": results
    }
    joblib.dump(bundle, save_file)
    st.success(f"Pelatihan selesai dan model tersimpan di {save_file}")
    return bundle

def load_bundle(file=MODEL_FILE):
    bundle = joblib.load(file)
    if 'rf' not in bundle or 'svc' not in bundle:
        if 'model' in bundle:
            model = bundle['model']
            try:
                named = getattr(model, "named_estimators_", None)
                if named and 'rf' in named and 'svc' in named:
                    bundle['rf'] = named['rf']
                    bundle['svc'] = named['svc']
                    bundle['voting'] = model
                else:
                    ests = getattr(model, "estimators_", None)
                    if ests and len(ests) >= 2:
                        bundle['rf'] = ests[0]
                        bundle['svc'] = ests[1]
                        bundle['voting'] = model
            except Exception:
                pass
    return bundle

dataset_path = find_dataset()
if dataset_path:
    st.write("Dataset ditemukan di:", dataset_path)
else:
    st.warning("Dataset tidak ditemukan. Upload file 'camera_dataset.csv' ke repo atau drive atau sesuaikan path.")
    st.stop()

if os.path.exists(MODEL_FILE):
    bundle = load_bundle(MODEL_FILE)
    st.success("Model ditemukan dan dimuat dari disk.")
else:
    df, X, y = load_and_prepare(dataset_path)
    bundle = train_all(X, y)

rf = bundle.get("rf", None)
svc = bundle.get("svc", None)
voting = bundle.get("voting", None)
scaler = bundle.get("scaler", None)
feature_names = bundle.get("feature_names", None)
evals = bundle.get("eval", None)

if scaler is None:
    st.error("Scaler tidak ditemukan dalam model bundle. Harap latih ulang menggunakan train script.")
    st.stop()
if feature_names is None:
    st.error("Feature names tidak ditemukan dalam model bundle. Harap latih ulang menggunakan train script.")
    st.stop()

st.sidebar.header("Pilih Model untuk Prediksi")
model_choice = st.sidebar.selectbox("Model", ("Voting (ensemble)", "RandomForest", "SVC"))

if evals:
    st.sidebar.subheader("Perbandingan Akurasi (test set)")
    acc_table = {name: round(info["accuracy"], 4) for name, info in evals.items()}
    st.sidebar.write(acc_table)
else:
    st.sidebar.info("Evaluasi tidak tersedia (bundle lama).")

st.subheader("Informasi fitur yang dipakai (urutan)")
st.write(feature_names)

st.subheader("Prediksi 1 Kamera (masukkan nilai numerik untuk setiap fitur)")
st.write("Jika tidak tahu, isi 0.")

inputs = []
cols = st.columns(2)
for i, fname in enumerate(feature_names):
    c = cols[i % 2]
    val = c.text_input(fname, value="0")
    inputs.append(val)

if st.button("Predict"):
    model_map = {
        "Voting (ensemble)": voting,
        "RandomForest": rf,
        "SVC": svc
    }
    chosen_model = model_map.get(model_choice)
    if chosen_model is None:
        st.error("Model yang dipilih tidak tersedia. Pastikan bundle berisi rf, svc, dan voting.")
        st.stop()

    try:
        vals = [float(x) for x in inputs]
    except Exception:
        st.error("Pastikan semua input berupa angka (contoh: 12.0).")
        st.stop()

    arr = np.array(vals).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred = chosen_model.predict(arr_scaled)[0]
    prob = chosen_model.predict_proba(arr_scaled)[0,1] if hasattr(chosen_model, "predict_proba") else None

    if pred == 1:
        st.success("Prediksi: PREMIUM")
    else:
        st.info("Prediksi: NON-PREMIUM")
    if prob is not None:
        st.write(f"Probabilitas premium: {prob:.3f}")

if st.checkbox("Tampilkan evaluasi lengkap (classification report & confusion matrix)"):
    if evals:
        for name, info in evals.items():
            st.markdown(f"### {name}")
            st.write("Accuracy:", info["accuracy"])
            st.text("Classification report:\n" + str(info["report"]))
            st.text("Confusion matrix:\n" + str(info["cm"]))
    else:
        st.info("Evaluasi tidak tersedia dalam model bundle.")

st.markdown("---")