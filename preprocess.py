import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
#---------- PRE‑PROCESS + AUGMENT ----------
def preprocess_and_augment(img_bgr, size=(224, 224)):
    """
    • Konversi ke grayscale → resize → blur
    • Hasilkan 4 varian: ori, flip‑horizontal, rotasi 15°, Gaussian‑noise
    """
    img_gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resize = cv2.resize(img_gray, size, interpolation=cv2.INTER_AREA)
    img_blur   = cv2.GaussianBlur(img_resize, (5, 5), 0)

    aug_imgs = [img_blur]                                # original
    aug_imgs.append(cv2.flip(img_blur, 1))               # flip
    h, w = img_blur.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 15, 1)
    aug_imgs.append(cv2.warpAffine(img_blur, M, (w, h))) # rotasi
    noise = np.random.normal(0, 25, img_blur.shape).astype(np.int16)
    noisy = np.clip(img_blur.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    aug_imgs.append(noisy)                               # noise
    return aug_imgs

def extract_glcm_features(image_array):
    """
    Fungsi untuk melakukan preprocessing dan ekstraksi fitur GLCM 
    dari sebuah gambar yang di-upload.
    """
    # 2. Ekstraksi Fitur GLCM
    # Pastikan gambar sudah dalam format integer 8-bit
    img_norm = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(img_norm, distances=[1], angles=angles, levels=256, symmetric=True, normed=True)
    
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
def extract_glcm(img_bgr):
    """
    • Buat 4 varian hasil augmentasi
    • Hitung fitur GLCM untuk tiap varian
    • Kembalikan rata‑ratanya ⇢ vektor panjang 10
    """
    variants = preprocess_and_augment(img_bgr)      # 4 citra
    feats    = np.vstack([extract_glcm_features(v) for v in variants])
    return feats.mean(axis=0)                       # shape (10,)