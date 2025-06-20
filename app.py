import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import plotly.figure_factory as ff
import numpy as np

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Dashboard Prediksi Kesehatan",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI UNTUK MEMUAT DAN MEMPROSES DATA ---
@st.cache_data
def load_data():
    """Memuat, membersihkan, dan mempersiapkan data dari file CSV."""
    data = pd.read_csv('healthcare_dataset.csv')
    
    # Menghapus kolom yang tidak relevan
    data.drop(['Name', 'Date of Admission', 'Discharge Date', 'Doctor',
               'Hospital', 'Insurance Provider', 'Room Number', 'Medication'], axis=1, inplace=True)
    
    # Mengisi nilai null dengan metode forward fill
    data.ffill(inplace=True)

    # Label encoding untuk kolom kategorikal
    encoders = {}
    categorical_cols = ['Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Test Results']
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    return data, encoders

# Memuat data dan encoder
data, encoders = load_data()

# Memisahkan fitur dan target
X = data.drop('Medical Condition', axis=1)
y = data['Medical Condition']

# Split data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- MELATIH MODEL ---
@st.cache_resource
def train_model(X_train, y_train):
    """Melatih model Random Forest Classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

# Melatih model
model = train_model(X_train, y_train)

# --- SIDEBAR ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Dashboard Prediksi", "Laporan Performa Model"])

# =================================================================================
# Halaman 1: Laporan Performa Model
# =================================================================================
if page == "Laporan Performa Model":
    st.title("Laporan Performa Model Prediksi")
    st.markdown("Halaman ini menampilkan metrik evaluasi dari model *Random Forest Classifier* yang telah dilatih.")

    # Prediksi pada data test
    y_pred = model.predict(X_test)

    # Metrik evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=encoders['Medical Condition'].classes_, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    # Tampilkan metrik utama
    st.header("Metrik Utama")
    col1, col2 = st.columns(2)
    col1.metric("Akurasi Model", f"{accuracy:.2%}")

    # Laporan klasifikasi
    st.header("Laporan Klasifikasi")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(2))
    st.info("""
    **Penjelasan Metrik:**
    - **Precision**: Ketepatan prediksi positif.
    - **Recall**: Kelengkapan deteksi kondisi aktual.
    - **F1-score**: Rata-rata harmonik dari Precision dan Recall.
    - **Support**: Jumlah data aktual untuk tiap kelas.
    """)

    # Confusion matrix
    st.header("Confusion Matrix")
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=list(encoders['Medical Condition'].classes_),
        y=list(encoders['Medical Condition'].classes_),
        colorscale='Viridis',
        showscale=True
    )
    fig.update_layout(
        title_text='<b>Confusion Matrix</b>',
        xaxis_title='Prediksi Model',
        yaxis_title='Label Aktual'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("Confusion Matrix menunjukkan perbandingan hasil prediksi dengan label aktual. Diagonal = prediksi benar.")

# =================================================================================
# Halaman 2: Dashboard Prediksi Interaktif
# =================================================================================
elif page == "Dashboard Prediksi":
    st.title("Dashboard Prediksi Kondisi Medis")
    st.markdown("Masukkan data pasien di bawah ini untuk mendapatkan prediksi kondisi medis menggunakan model yang telah dilatih.")

    with st.form("prediction_form"):
        st.header("Input Data Pasien")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Usia", min_value=1, max_value=100, value=55, step=1)
            gender_input = st.selectbox("Jenis Kelamin", options=encoders['Gender'].classes_)
            blood_type_input = st.selectbox("Tipe Darah", options=encoders['Blood Type'].classes_)

        with col2:
            admission_type_input = st.selectbox("Tipe Pendaftaran", options=encoders['Admission Type'].classes_)
            test_results_input = st.selectbox("Hasil Tes", options=encoders['Test Results'].classes_)
            billing_amount = st.number_input("Jumlah Tagihan (Billing Amount)", min_value=0.0, value=5000.0, step=100.0)

        submit_button = st.form_submit_button(label="Dapatkan Prediksi")

    if submit_button:
        input_data = {
            'Age': [age],
            'Gender': [encoders['Gender'].transform([gender_input])[0]],
            'Blood Type': [encoders['Blood Type'].transform([blood_type_input])[0]],
            'Admission Type': [encoders['Admission Type'].transform([admission_type_input])[0]],
            'Test Results': [encoders['Test Results'].transform([test_results_input])[0]],
            'Billing Amount': [billing_amount]
        }

        input_df = pd.DataFrame(input_data)

        prediction_encoded = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)

        prediction_label = encoders['Medical Condition'].inverse_transform([prediction_encoded])[0]

        st.header("Hasil Prediksi")
        st.success(f"**Prediksi Kondisi Medis:** {prediction_label}")

        st.subheader("Probabilitas Prediksi untuk Setiap Kondisi")
        proba_df = pd.DataFrame(
            prediction_proba,
            columns=encoders['Medical Condition'].classes_,
            index=['Probabilitas']
        ).transpose().sort_values(by='Probabilitas', ascending=False)

        st.dataframe(proba_df.style.format("{:.2%}").background_gradient(cmap='viridis'))
        st.info("Tabel di atas menunjukkan tingkat keyakinan model untuk setiap kemungkinan kondisi medis.")
