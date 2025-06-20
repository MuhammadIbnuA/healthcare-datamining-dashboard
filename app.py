# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import glob
import os
import plotly.figure_factory as ff

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Dashboard Komparasi Model Prediksi",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Fungsi Helper dan Caching ---

@st.cache_data
def find_available_models():
    """Mencari semua model .joblib dan report .json yang tersedia di folder."""
    model_files = glob.glob('*_model.joblib')
    model_names = {os.path.basename(f).replace('_model.joblib', ''): f for f in model_files}
    
    report_files = glob.glob('*_report.json')
    report_names = {os.path.basename(f).replace('_report.json', ''): f for f in report_files}
    
    # Hanya kembalikan model yang memiliki file model dan report
    available_models = {name: {'model_path': model_path, 'report_path': report_names.get(name)} 
                        for name, model_path in model_names.items() if name in report_names}
    return available_models

@st.cache_resource
def load_model(model_path):
    """Memuat model dari file path."""
    return joblib.load(model_path)

@st.cache_data
def load_report(report_path):
    """Memuat laporan dari file path."""
    with open(report_path, 'r') as f:
        return json.load(f)

@st.cache_data
def get_ui_values():
    """Memuat nilai unik untuk dropdown UI dari dataset asli."""
    df = pd.read_csv('healthcare_dataset.csv')
    return {
        'Gender': df['Gender'].unique(),
        'Blood Type': df['Blood Type'].unique(),
        'Admission Type': df['Admission Type'].unique(),
        'Test Results': df['Test Results'].unique()
    }

# --- Logika Utama Aplikasi ---

available_models_info = find_available_models()
ui_values = get_ui_values()

if not available_models_info:
    st.error("Tidak ditemukan file model dan laporan yang cocok (contoh: 'svm_model.joblib' dan 'svm_report.json'). "
             "Pastikan Anda sudah menjalankan skrip training terlebih dahulu.")
    st.stop()

# --- Sidebar ---
st.sidebar.title("⚙️ Pengaturan Model")
model_selection_key = st.sidebar.selectbox(
    "Pilih Model untuk Digunakan:",
    options=list(available_models_info.keys()),
    format_func=lambda x: x.replace('_', ' ').title() # Membuat nama lebih mudah dibaca
)

# Memuat model dan laporan yang dipilih
selected_model_info = available_models_info[model_selection_key]
model = load_model(selected_model_info['model_path'])
report = load_report(selected_model_info['report_path'])

st.sidebar.success(f"Model **{report['model_name']}** berhasil dimuat.")
st.sidebar.metric("Akurasi Test Model Ini", f"{report['final_test_accuracy']:.2%}")

st.sidebar.title("📖 Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["🔮 Prediksi Interaktif", "📊 Laporan Training"])


# --- Tampilan Halaman ---

if page == "🔮 Prediksi Interaktif":
    st.title(f"🔮 Prediksi Menggunakan Model: {report['model_name']}")
    st.markdown("Masukkan data pasien di bawah ini untuk mendapatkan prediksi dari model yang aktif.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Usia", 1, 100, 55)
            gender = st.selectbox("Jenis Kelamin", ui_values['Gender'])
            blood_type = st.selectbox("Tipe Darah", ui_values['Blood Type'])
        with col2:
            admission_type = st.selectbox("Tipe Pendaftaran", ui_values['Admission Type'])
            test_results = st.selectbox("Hasil Tes", ui_values['Test Results'])
        
        submit_button = st.form_submit_button("Dapatkan Prediksi")

    if submit_button:
        input_df = pd.DataFrame([{
            'Age': age, 'Gender': gender, 'Blood Type': blood_type,
            'Admission Type': admission_type, 'Test Results': test_results
        }])
        
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)

        st.header("Hasil Prediksi")
        st.success(f"**Prediksi Kondisi Medis:** {prediction}")

        st.subheader("Tingkat Keyakinan Model:")
        proba_df = pd.DataFrame(proba, columns=model.classes_, index=['Probabilitas']).transpose()
        proba_df = proba_df.sort_values('Probabilitas', ascending=False)
        st.dataframe(proba_df.style.format("{:.2%}").background_gradient(cmap='Greens'))


elif page == "📊 Laporan Training":
    st.title(f"📊 Laporan Performa Model: {report['model_name']}")
    
    st.header("Performa Umum")
    col1, col2 = st.columns(2)
    col1.metric("Akurasi Cross-Validation Terbaik", f"{report['best_cv_accuracy']:.2%}")
    col2.metric("Akurasi Final pada Data Tes", f"{report['final_test_accuracy']:.2%}")

    st.header("Konfigurasi Model Terbaik")
    # Mengubah format parameter untuk tampilan yang lebih baik
    params = report.get('best_parameters', {})
    clean_params = {k.split('__')[-1]: v for k, v in params.items()}
    st.json(clean_params)

    st.header("Laporan Klasifikasi Rinci")
    class_report_df = pd.DataFrame(report['classification_report']).transpose()
    st.dataframe(class_report_df.round(3))

    # Cek apakah ada data confusion matrix untuk ditampilkan
    if 'confusion_matrix' in report and 'target_names' in report.get('classification_report', {}):
        st.header("Confusion Matrix")
        cm = np.array(report['confusion_matrix'])
        target_names = list(report['classification_report'].keys())[:-3] # Ambil nama kelas dari laporan
        fig = ff.create_annotated_heatmap(z=cm, x=target_names, y=target_names, colorscale='Blues')
        fig.update_layout(xaxis_title='Prediksi Model', yaxis_title='Label Aktual')
        st.plotly_chart(fig, use_container_width=True)