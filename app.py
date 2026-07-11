import streamlit as st
import numpy as np
from nptdms import TdmsFile
from tensorflow.keras.models import load_model

# 1. Konfigurasi Halaman (Wide Layout)
st.set_page_config(page_title="Dasbor Pemeliharaan PMSM", page_icon="⚙️", layout="wide")

# 2. Sidebar untuk Konteks Sistem
with st.sidebar:
    st.header("Spesifikasi Sistem")
    st.info("""
    - **Algoritma:** 1D-CNN
    - **Target Deteksi:** Stator Faults
    - **Sensor Input:** Akselerometer Sumbu-Z
    - **Kapasitas Mesin:** 1.0 kW
    """)
    st.markdown("---")
    st.write("Sistem ini mendiagnosis kondisi isolasi kumparan motor secara real-time berdasarkan matriks sinyal mekanis.")

# 3. Header Utama
st.title("Sistem Pemeliharaan Prediktif PMSM")
st.markdown("---")

# 4. Inisialisasi Model
@st.cache_resource
def load_pmsm_model():
    model = load_model('pmsm_vibration_model.h5')
    labels = np.load('label_classes.npy', allow_pickle=True)
    return model, labels

try:
    model, labels = load_pmsm_model()
except Exception as e:
    st.error("Gagal memuat model. Pastikan file .h5 dan .npy berada di direktori yang sama dengan app.py.")
    st.stop()

WINDOW_SIZE = 2048

# 5. Komponen Unggah File
uploaded_file = st.file_uploader("Unggah log data getaran (.tdms)", type=['tdms'])

if uploaded_file is not None:
    with st.spinner('Mengekstraksi matriks getaran...'):
        try:
            tdms_file = TdmsFile.read(uploaded_file)
            
            vibration_signal = None
            for group in tdms_file.groups():
                channels = group.channels()
                if len(channels) > 0:
                    vibration_signal = channels[0].data
                    break
            
            if vibration_signal is not None and len(vibration_signal) >= WINDOW_SIZE:
                segment = vibration_signal[:WINDOW_SIZE]
                X_input = segment.reshape((1, WINDOW_SIZE, 1))
                
                # Prediksi
                prediction = model.predict(X_input)
                predicted_idx = np.argmax(prediction, axis=1)[0]
                predicted_class = labels[predicted_idx]
                confidence = np.max(prediction) * 100
                
                # 6. Tata Letak Hasil Diagnosis (Kolom)
                st.subheader("Hasil Diagnosis Sistem")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if predicted_class == 'normal':
                        st.success("STATUS AMAN: Tidak ditemukan anomali pada lilitan stator.")
                    elif predicted_class == 'coil_short':
                        st.warning("PERINGATAN DINI: Terdeteksi korsleting antar-kumparan (Coil-to-Coil Short). Lakukan inspeksi isolasi utama.")
                    elif predicted_class == 'interturn_short':
                        st.error("STATUS KRITIS: Terdeteksi korsleting antar-lilitan (Inter-turn Short). Risiko degradasi termal lokal.")
                
                with col2:
                    st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{confidence:.2f}%")
                
                # 7. Sistem Tab untuk Visualisasi Lanjutan
                st.markdown("---")
                tab1, tab2 = st.tabs(["Visualisasi Gelombang", "Metrik Data Mentah"])
                
                with tab1:
                    st.line_chart(segment)
                    
                with tab2:
                    st.write("Statistik Sinyal Segmen (2048 titik):")
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    stat_col1.metric("Max Amplitudo", f"{np.max(segment):.4f}")
                    stat_col2.metric("Min Amplitudo", f"{np.min(segment):.4f}")
                    stat_col3.metric("Rata-rata", f"{np.mean(segment):.4f}")
                    
            else:
                st.error("Gagal: File TDMS tidak valid atau tidak memiliki cukup data titik.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan pemrosesan: {e}")