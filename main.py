import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(
    page_title="PyImageDesk Web",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #1E293B;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        color: #64748B;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">PyImageDesk: Web Edition</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Integrated Image Processing and Educational Analysis Platform</p>', unsafe_allow_html=True)

# Sidebar - Image Upload
st.sidebar.header("📁 Image Source")
uploaded_file = st.sidebar.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

# Default image fallback if nothing uploaded
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
else:
    image_rgb = np.zeros((400, 600, 3), dtype=np.uint8)
    image_rgb[:] = (240, 240, 240)
    cv2.putText(image_rgb, "Please Upload an Image", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

# Sidebar - Pipeline Controls
st.sidebar.header("⚙️ Processing Pipeline")
filter_category = st.sidebar.selectbox(
    "Select Category",
    ["Smoothing (Blur)", "Edge Detection", "Thresholding", "Morphology", "Enhancement"]
)

processed_image = image_rgb.copy()

if uploaded_file is not None:
    if filter_category == "Smoothing (Blur)":
        blur_type = st.sidebar.radio("Blur Method", ["Gaussian Blur", "Median Blur"])
        ksize = st.sidebar.slider("Kernel Size (Odd numbers)", 1, 31, 5, step=2)
        if blur_type == "Gaussian Blur":
            processed_image = cv2.GaussianBlur(image_rgb, (ksize, ksize), 0)
        else:
            processed_image = cv2.medianBlur(image_rgb, ksize)

    elif filter_category == "Edge Detection":
        edge_method = st.sidebar.selectbox("Operator", ["Canny Edge", "Sobel X", "Scharr X"])
        if edge_method == "Canny Edge":
            t1 = st.sidebar.slider("Threshold 1", 0, 255, 100)
            t2 = st.sidebar.slider("Threshold 2", 0, 255, 200)
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, t1, t2)
            processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        elif edge_method == "Sobel X":
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel = cv2.convertScaleAbs(sobel)
            processed_image = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
        elif edge_method == "Scharr X":
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            scharr = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
            scharr = cv2.convertScaleAbs(scharr)
            processed_image = cv2.cvtColor(scharr, cv2.COLOR_GRAY2RGB)

    elif filter_category == "Thresholding":
        thresh_type = st.sidebar.selectbox("Method", ["Binary Threshold", "Otsu Threshold"])
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        if thresh_type == "Binary Threshold":
            val = st.sidebar.slider("Threshold Value", 0, 255, 127)
            _, thresh = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
            processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        else:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    elif filter_category == "Morphology":
        op = st.sidebar.selectbox("Operation", ["Dilation", "Erosion"])
        ksize = st.sidebar.slider("Kernel Size", 1, 15, 3)
        kernel = np.ones((ksize, ksize), np.uint8)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        if op == "Dilation":
            res = cv2.dilate(gray, kernel, iterations=1)
        else:
            res = cv2.erode(gray, kernel, iterations=1)
        processed_image = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)

    elif filter_category == "Enhancement":
        enh = st.sidebar.selectbox("Technique", ["Sharpen", "Histogram Equalization"])
        if enh == "Sharpen":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            processed_image = cv2.filter2D(image_rgb, -1, kernel)
        else:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            eq = cv2.equalizeHist(gray)
            processed_image = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)

# Main Display: Side by Side
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original Image")
    st.image(image_rgb, use_container_width=True)

with col2:
    st.subheader("Processed Output")
    st.image(processed_image, use_container_width=True)

# Analytics Section
if uploaded_file is not None:
    st.markdown("---")
    st.subheader("📊 Spectral & Intensity Analytics")
    
    tab1, tab2 = st.tabs(["RGB Histograms", "Fast Fourier Transform (FFT)"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(8, 3))
        colors = ('r', 'g', 'b')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
            ax.plot(hist, color=col)
            ax.set_xlim([0, 256])
        ax.set_title("Color Channel Intensity Distribution")
        st.pyplot(fig)

    with tab2:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
        
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(magnitude_spectrum, cmap='gray')
        ax.set_title("2D Magnitude Spectrum (FFT)")
        ax.axis('off')
        st.pyplot(fig)
      
