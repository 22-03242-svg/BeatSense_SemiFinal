# streamlit_ecg_app.py
import streamlit as st
import os
import wfdb
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="ECG Analysis App", layout="wide")

# --- Upload ECG files ---
st.title("ECG Analysis & Tachycardia Detection")
st.write("Upload your ECG record files (.dat, .hea, .atr)")

uploaded_files = st.file_uploader("Choose ECG files", type=['dat','hea','atr'], accept_multiple_files=True)
data_dir = "ecg_data"
os.makedirs(data_dir, exist_ok=True)

file_list = []
for uploaded_file in uploaded_files:
    path = os.path.join(data_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_list.append(uploaded_file.name)

if len(file_list) == 0:
    st.warning("Please upload at least one ECG file to proceed.")
    st.stop()

# --- Select file to process ---
chosen_file = st.selectbox("Select a file to analyze", file_list)
chosen_path = os.path.join(data_dir, chosen_file)
st.write(f"Processing file: {chosen_file}")

# --- FUNCTIONS ---
def bandpass(sig, fs, low=0.5, high=40):
    b, a = butter(3, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, sig)

def pan_tompkins_detector(signal, fs):
    b, a = butter(3, [5/(fs/2), 15/(fs/2)], btype='band')
    filtered_ecg = filtfilt(b, a, signal)
    diff_signal = np.ediff1d(filtered_ecg, to_end=0)
    squared = diff_signal ** 2
    window_size = int(0.150 * fs)
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    distance = int(0.25 * fs)
    height = np.mean(integrated) * 1.2
    peaks, _ = find_peaks(integrated, distance=distance, height=height)
    refined_peaks = []
    search_radius = int(0.05 * fs)
    for p in peaks:
        start = max(p - search_radius, 0)
        end = min(p + search_radius, len(signal))
        local_max = np.argmax(signal[start:end]) + start
        refined_peaks.append(local_max)
    return np.unique(refined_peaks)

def extract_beats(signal, r_peaks, fs, window_ms=700, resample_len=100):
    half = int((window_ms / 1000) * fs // 2)
    beats = []
    for r in r_peaks:
        if r - half < 0 or r + half >= len(signal):
            continue
        beat = signal[r - half:r + half]
        beats.append(resample(beat, resample_len))
    return np.array(beats)

def extract_features(beats, rr_intervals):
    features = []
    for i, beat in enumerate(beats):
        rr = rr_intervals[i] if i < len(rr_intervals) else rr_intervals[-1]
        features.append([
            np.mean(beat),
            np.std(beat),
            np.min(beat),
            np.max(beat),
            rr,
            np.median(beat),
            np.percentile(beat, 25),
            np.percentile(beat, 75),
            np.sum(beat**2),
            len(beat)
        ])
    return np.array(features)

def is_irregular(rr_segment, threshold=0.12):
    return np.std(rr_segment) > threshold

def classify_tachycardia_regular(beat_seq):
    if any(b == "V" for b in beat_seq):
        return "Ventricular Tachycardia"
    if any(b == "A" for b in beat_seq):
        return "Atrial Flutter"
    if any(b in ["L", "R"] for b in beat_seq):
        return "Supraventricular Tachycardia"
    return "Supraventricular Tachycardia"

def classify_tachycardia_irregular(beat_seq):
    if any(b == "F" for b in beat_seq):
        return "Atrial Fibrillation"
    if any(b == "V" for b in beat_seq):
        return "Ventricular Fibrillation"
    return "Atrial Fibrillation"

label_map = {"N":0, "L":1, "R":2, "V":3, "A":4, "F":5}
max_duration_sec = 120

# --- Load ECG data ---
try:
    base_name = chosen_file[:-4] if chosen_file.lower().endswith(".hea") else chosen_file
    record_path = os.path.join(data_dir, base_name)
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")
    signal = record.p_signal[:,0]
    fs = record.fs
    max_samples = int(max_duration_sec * fs)
    signal = signal[:max_samples]
    r_peaks = ann.sample
    labels = np.array(ann.symbol)
    valid_idx = np.where(r_peaks < max_samples)[0]
    r_peaks = r_peaks[valid_idx]
    labels = labels[valid_idx]
except Exception as e:
    st.error(f"Failed to load record: {e}")
    st.stop()

# --- Preprocessing ---
signal_f = bandpass(signal, fs)
beats = extract_beats(signal_f, r_peaks, fs)
rr = np.diff(r_peaks)/fs
rr = np.append(rr, rr[-1])
beat_features = extract_features(beats, rr)
y_beats = np.array([label_map.get(l, 0) for l in labels[:len(beat_features)]])

# --- Beat-level classification ---
X_train, X_test, y_train, y_test = train_test_split(beat_features, y_beats, test_size=0.2, random_state=42)
clf_beats = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
clf_beats.fit(X_train, y_train)
pred_beats = clf_beats.predict(X_test)

st.subheader("Beat Classification Report")
st.text(classification_report(y_test, pred_beats, zero_division=0))

st.subheader("Confusion Matrix")
st.write(confusion_matrix(y_test, pred_beats))

# --- Sequence-level labeling ---
beat_hr_labels = ["Bradycardia" if 60/rrv < 60 else "Tachycardia" if 60/rrv > 100 else "Normal" for rrv in rr]

seq_len = 25
seq_step = 5
seq_labels = []
tachy_results = []

for i in range(0, len(rr)-seq_len, seq_step):
    seq_rr = rr[i:i+seq_len]
    avg_hr = 60/np.mean(seq_rr)
    if avg_hr < 60:
        seq_labels.append(0)
        tachy_results.append("Not Tachycardia")
    elif avg_hr > 100:
        seq_labels.append(2)
        seq_beats = labels[i:i+seq_len]
        if is_irregular(seq_rr):
            subtype = classify_tachycardia_irregular(seq_beats)
        else:
            subtype = classify_tachycardia_regular(seq_beats)
        tachy_results.append(subtype)
    else:
        seq_labels.append(1)
        tachy_results.append("Not Tachycardia")

# --- Overall Summary ---
overall_summary = {"Bradycardia":0,"Normal":0,"Tachycardia":0,"AFib":0,"VT":0,"SVT":0,"AFlutter":0,"Other Tachy":0}
for i,label in enumerate(seq_labels):
    if label==0:
        overall_summary["Bradycardia"] += 1
    elif label==1:
        overall_summary["Normal"] +=1
    else:
        overall_summary["Tachycardia"] +=1
        subtype = tachy_results[i]
        if subtype=="Atrial Fibrillation":
            overall_summary["AFib"]+=1
        elif subtype=="Ventricular Tachycardia":
            overall_summary["VT"]+=1
        elif subtype=="Supraventricular Tachycardia":
            overall_summary["SVT"]+=1
        elif subtype=="Atrial Flutter":
            overall_summary["AFlutter"]+=1
        else:
            overall_summary["Other Tachy"]+=1

st.subheader("Overall Rhythm Summary")
summary_df = pd.DataFrame.from_dict(overall_summary, orient='index', columns=['Sequences'])
summary_df['% of Total'] = summary_df['Sequences']/summary_df['Sequences'].sum()*100
st.dataframe(summary_df)

# --- ECG Plots ---
st.subheader("ECG Signal with R-peaks")
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(signal, label='ECG Signal')
ax.scatter(r_peaks, signal[r_peaks], color='red', label='Detected R-peaks')
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")
ax.legend()
st.pyplot(fig)

st.subheader("First 20 Beats HR and Rhythm")
num_beats_to_plot = min(20, len(r_peaks))
first_r = r_peaks[:num_beats_to_plot]
rr_intervals_20 = np.diff(first_r)/fs
heart_rate_20 = 60/rr_intervals_20

fig2, ax2 = plt.subplots(figsize=(12,4))
ax2.plot(heart_rate_20, marker='o', linestyle='-', alpha=0.8)
ax2.set_xlabel("Beat Number")
ax2.set_ylabel("Heart Rate (bpm)")
ax2.set_title("Heart Rate Trend (First 20 Beats)")
ax2.grid(True)
st.pyplot(fig2)
