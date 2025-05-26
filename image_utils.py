import cv2
import numpy as np

# Fungsi untuk mengubah citra berwarna menjadi grayscale
def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Fungsi untuk menerapkan Gaussian blur
def gaussian_blur(image, ksize=(3,3)):
    return cv2.GaussianBlur(image, ksize, 0)

# Fungsi untuk menerapkan thresholding Otsu pada citra grayscale
def apply_otsu_threshold(gray):
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# Fungsi untuk mendeteksi tepi menggunakan operator Prewitt
def prewitt_edge(image):
    # Kernel Prewitt untuk arah x dan y
    kernelx = np.array([[ -1,  0,  1],
                        [ -1,  0,  1],
                        [ -1,  0,  1]])
    
    kernely = np.array([[ 1,  1,  1],
                        [ 0,  0,  0],
                        [-1, -1, -1]])
    
    # Terapkan filter Prewitt ke citra
    x = cv2.filter2D(image, -1, kernelx)
    y = cv2.filter2D(image, -1, kernely)
    
    # Gabungkan hasil gradien
    edge = cv2.magnitude(x.astype(np.float32), y.astype(np.float32))
    edge = np.uint8(np.clip(edge, 0, 255))

    return edge

# Fungsi untuk mendeteksi tepi menggunakan operator Sobel
def sobel_edge(image):
    # Hitung gradien arah x dan y
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Konversi gradien ke nilai absolut dan 8-bit
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # Gabungkan gradien x dan y dengan bobot yang sama
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# Fungsi untuk melakukan histogram equalization
def histogram_equalization(image):
    return cv2.equalizeHist(image)


# Fungsi untuk mengompresi citra grayscale menggunakan kuantisasi
def quantize_compression(image, levels=16):
    # Konversi ke grayscale jika gambar berwarna
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Hitung ukuran langkah kuantisasi
    step = 256 // levels
    
    # Lakukan kuantisasi (mengelompokkan nilai piksel)
    compressed = (image // step) * step
    
    return compressed.astype(np.uint8)

