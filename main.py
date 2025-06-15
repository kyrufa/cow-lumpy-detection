import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import joblib
from skimage.feature import graycomatrix, graycoprops

# --- KONFIGURASI DAN FUNGSI BANTU ---

# Set judul halaman dan ikon
st.set_page_config(page_title="Deteksi Penyakit Kulit Sapi", page_icon="🐄")

# Fungsi untuk memuat model dan scaler (menggunakan cache resource)
@st.cache_resource
def load_model_and_scaler():
    """Memuat model KNN dan scaler yang sudah dilatih."""
    try:
        model = joblib.load('model_knn_sapi.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("File model 'model_knn_sapi.pkl' atau 'scaler.pkl' tidak ditemukan.")
        st.info("Pastikan Anda sudah menyimpan model dan scaler dari notebook Anda di folder yang sama dengan app.py")
        return None, None

# ---------- PRE‑PROCESS + AUGMENT ----------
def preprocess_and_augment(img_array, size=(256, 256)):
    """
    • Konversi ke grayscale → resize → blur
    • Hasilkan 4 varian: ori, flip‑horizontal, rotasi 15°, Gaussian‑noise
    """
    # FIX: Handle color (RGB from PIL) vs grayscale images robustly
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array # It's already grayscale

    st.image(img_gray, caption='Gambar grayscale', use_column_width=True)

    img_resize = cv2.resize(img_gray, size)
    st.image(img_resize, caption='Gambar setelah resize', use_column_width=True)

    img_blur   = cv2.GaussianBlur(img_resize, (5, 5), 0)
    st.image(img_blur, caption='Gambar setelah Gaussian Blur', use_column_width=True)

    img_norm = cv2.normalize(img_blur, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    st.image(img_norm, caption='Gambar setelah normalisasi', use_column_width=True)

    # Augmentasi citra
    aug_imgs = [img_norm]                                # original
    aug_imgs.append(cv2.flip(img_norm, 1))               # flip
    st.image(aug_imgs[1], caption='Gambar setelah flip horizontal', use_column_width=True)

    h, w = img_norm.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 15, 1)
    aug_imgs.append(cv2.warpAffine(img_norm, M, (w, h))) # rotasi
    st.image(aug_imgs[2], caption='Gambar setelah rotasi 15 derajat', use_column_width=True)

    noise = np.random.normal(0, 25, img_norm.shape).astype(np.int16)
    noisy = np.clip(img_norm.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    aug_imgs.append(noisy)                               # noise
    st.image(aug_imgs[3], caption='Gambar setelah penambahan noise Gaussian', use_column_width=True)
    return aug_imgs

def extract_glcm_features(image_array):
    """
    Fungsi untuk melakukan preprocessing dan ekstraksi fitur GLCM 
    dari sebuah gambar yang di-upload.
    """
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(image_array, distances=[1], angles=angles, levels=256, symmetric=True, normed=True)
    
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    asm = np.mean(graycoprops(glcm, 'ASM'))
    IDM = np.mean([1 / (1 + contrast)])  # Inverse Difference Moment
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))  # Hindari log(0)
    mean = np.mean(glcm)
    median = np.median(glcm)

    # Kumpulkan semua fitur ke dalam satu array
    features = np.array([contrast, dissimilarity, homogeneity, energy, correlation, asm, IDM, entropy, mean, median])
    
    return features

#----- GLCM FEATURES -----
def extract_glcm(img_array):
    """
    • Buat 4 varian hasil augmentasi
    • Hitung fitur GLCM untuk tiap varian
    • Kembalikan rata‑ratanya ⇢ vektor panjang 10
    """
    # FIX: Pass img_array instead of misleading img_bgr
    variants = preprocess_and_augment(img_array)      # 4 citra
    feats    = np.vstack([extract_glcm_features(v) for v in variants])
    return feats.mean(axis=0)                       # shape (10,)

# --- ANTARMUKA APLIKASI STREAMLIT ---

# Judul Utama Aplikasi
st.title("🐄 Aplikasi Deteksi Penyakit Kulit Sapi")
st.write("Aplikasi ini menggunakan model Machine Learning (KNN) dengan ekstraksi fitur GLCM untuk memprediksi penyakit kulit pada sapi berdasarkan citra.")

# Memuat model dan scaler di awal
model, scaler = load_model_and_scaler()

# Pindahkan uploader ke sidebar agar lebih rapi
st.sidebar.header("Upload Gambar Anda")
uploaded_file = st.sidebar.file_uploader("Pilih sebuah gambar kulit sapi...", type=["jpg", "png", "jpeg"])

if uploaded_file is None:
    st.info("Silakan upload gambar melalui panel di sebelah kiri untuk memulai deteksi.")


if uploaded_file is not None and model is not None:
    # Tampilkan gambar yang di-upload
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang di-upload', use_column_width=True)
    
    # Tombol untuk memulai proses deteksi
    if st.button("Analisis dan Deteksi Penyakit", type="primary"):
        with st.spinner('Sedang memproses gambar dan melakukan prediksi...'):
            # 1. Konversi gambar ke format yang bisa dibaca OpenCV
            img_array = np.array(image)

            # 2. Ekstraksi Fitur dari gambar yang di-upload
            new_features = extract_glcm(img_array)
            
            # 3. Normalisasi fitur baru menggunakan scaler yang sudah di-load
            # Reshape array fitur menjadi 2D karena scaler mengharapkan input 2D
            new_features_reshaped = new_features.reshape(1, -1)
            new_features_scaled = scaler.transform(new_features_reshaped)
            
            # 4. Lakukan Prediksi
            prediction = model.predict(new_features_scaled)
            prediction_proba = model.predict_proba(new_features_scaled)

            # --- PERBAIKAN DI SINI ---
            # FIX: Removed leading space from ' Sehat'
            class_names = {0: 'Lumpy Skin Disease (LSD)', 1: 'Sehat'} 
            predicted_class = class_names.get(prediction[0], "Kelas Tidak Dikenali")

        # 5. Tampilkan Hasil
        st.subheader("Hasil Deteksi")
        # FIX: Ensure comparison matches the key in class_names
        if predicted_class == 'Sehat':
            st.success(f"**Prediksi:** Sapi teridentifikasi **{predicted_class}**.")
        else:
            st.error(f"**Prediksi:** Terdeteksi penyakit **{predicted_class}**.")
        
        # Tampilkan probabilitas
        st.write("Probabilitas prediksi:")
        st.dataframe(pd.DataFrame(prediction_proba, columns=class_names.values()))
        
        # Tampilkan fitur yang diekstrak untuk transparansi
        with st.expander("Lihat Fitur GLCM yang Diekstrak dari Gambar"):
            features_df = pd.DataFrame([new_features], columns=['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM', 'IDM', 'Entropy', 'Mean', 'Median'])
            st.dataframe(features_df)

# Menampilkan data training sebagai referensi (opsional)
st.sidebar.markdown("---")
if st.sidebar.checkbox("Tampilkan Dataset Training (CSV)"):
    st.subheader("Dataset Fitur GLCM (Data Latih)")
    try:
        df = pd.read_csv("glcm_features.csv")
        st.dataframe(df)
    except FileNotFoundError:
        st.warning("File 'glcm_features.csv' tidak ditemukan.")
