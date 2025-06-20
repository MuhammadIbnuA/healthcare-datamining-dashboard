import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 5 algoritma ringan
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import plotly.figure_factory as ff

# --- Konfigurasi Streamlit ---
st.set_page_config(page_title="Prediksi Kesehatan", layout="wide")

# --- Sidebar ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Dashboard Prediksi", "Laporan Performa Model"])
model_choice = st.sidebar.selectbox("Pilih Algoritma ML", [
    "Logistic Regression", "Random Forest", "K-Nearest Neighbors", "Decision Tree", "Naive Bayes"
])

# --- Load & Preprocess Data ---
@st.cache_data
def load_data():
    data = pd.read_csv('healthcare_dataset.csv')
    data.drop(['Name', 'Date of Admission', 'Discharge Date', 'Doctor',
               'Hospital', 'Insurance Provider', 'Room Number', 'Medication'], axis=1, inplace=True)
    data.ffill(inplace=True)
    
    encoders = {}
    categorical_cols = ['Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Test Results']
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    return data, encoders

data, encoders = load_data()
X = data.drop('Medical Condition', axis=1)
y = data['Medical Condition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- Train Model ---
@st.cache_resource
def train_model(X_train, y_train, model_choice):
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced")
    elif model_choice == "K-Nearest Neighbors":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier(class_weight="balanced")
    elif model_choice == "Naive Bayes":
        model = GaussianNB()
    else:
        raise ValueError("Model tidak ditemukan")
    
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train, model_choice)

# --- Halaman: Dashboard Prediksi ---
if page == "Dashboard Prediksi":
    st.title("Dashboard Prediksi Kondisi Medis")

    with st.form("prediction_form"):
        st.subheader("Input Data Pasien")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Usia", min_value=1, max_value=100, value=55)
            gender = st.selectbox("Jenis Kelamin", encoders['Gender'].classes_)
            blood_type = st.selectbox("Tipe Darah", encoders['Blood Type'].classes_)

        with col2:
            admission_type = st.selectbox("Tipe Pendaftaran", encoders['Admission Type'].classes_)
            test_result = st.selectbox("Hasil Tes", encoders['Test Results'].classes_)
            billing_amount = st.number_input("Jumlah Tagihan (Billing Amount)", min_value=0.0, value=5000.0)

        submit = st.form_submit_button("Prediksi")

    if submit:
        input_data = {
            'Age': [age],
            'Gender': [encoders['Gender'].transform([gender])[0]],
            'Blood Type': [encoders['Blood Type'].transform([blood_type])[0]],
            'Admission Type': [encoders['Admission Type'].transform([admission_type])[0]],
            'Test Results': [encoders['Test Results'].transform([test_result])[0]],
            'Billing Amount': [billing_amount]
        }

        input_df = pd.DataFrame(input_data)
        input_df = input_df[X_train.columns]  # pastikan urutan kolom sesuai

        pred = model.predict(input_df)[0]
        pred_label = encoders['Medical Condition'].inverse_transform([pred])[0]
        proba = model.predict_proba(input_df)[0]

        st.subheader("Hasil Prediksi")
        st.success(f"Kondisi Medis: **{pred_label}**")

        st.subheader("Probabilitas")
        proba_df = pd.DataFrame({
            'Kondisi': encoders['Medical Condition'].classes_,
            'Probabilitas': proba
        }).sort_values(by='Probabilitas', ascending=False)

        st.dataframe(proba_df.style.format({'Probabilitas': "{:.2%}"}).background_gradient(cmap="Blues"))

# --- Halaman: Laporan Performa Model ---
elif page == "Laporan Performa Model":
    st.title("Laporan Performa Model")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, target_names=encoders['Medical Condition'].classes_)
    cm = confusion_matrix(y_test, y_pred)

    st.metric("Akurasi", f"{acc:.2%}")
    st.subheader("Laporan Klasifikasi")
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    st.subheader("Confusion Matrix")
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=encoders['Medical Condition'].classes_,
        y=encoders['Medical Condition'].classes_,
        colorscale="Viridis"
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Prediksi",
        yaxis_title="Aktual"
    )
    st.plotly_chart(fig)
