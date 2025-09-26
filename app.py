# app.py
import io
import math
from typing import Tuple, Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# ===========================
# ====== UI CONFIG ==========
# ===========================
st.set_page_config(
    page_title="Image Processing Playground",
    page_icon="🧪",
    layout="wide"
)
st.title("🧪 Image Processing Playground")
st.caption("Select a processing method, tune parameters, and compare before/after with histograms. Supports grayscale & color images.")

# ===========================
# ====== UTILITIES ==========
# ===========================

def _ensure_float01(img: np.ndarray) -> np.ndarray:
    """Convert to float32 in [0,1]."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    img = img.astype(np.float32)
    m, M = float(img.min()), float(img.max())
    if M <= m:
        return np.zeros_like(img, dtype=np.float32)
    return (img - m) / (M - m)

def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Clip to [0,1] and convert to uint8."""
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)

def _is_grayscale(arr: np.ndarray) -> bool:
    return (arr.ndim == 2) or (arr.ndim == 3 and arr.shape[2] == 1)

def _rgb_to_lab(rgb01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(_to_uint8(rgb01), cv2.COLOR_RGB2LAB).astype(np.float32)

def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lab_u8 = lab.copy()
    # Ensure valid ranges before conversion
    lab_u8[..., 0] = np.clip(lab_u8[..., 0], 0, 255)
    lab_u8[..., 1] = np.clip(lab_u8[..., 1], 0, 255)
    lab_u8[..., 2] = np.clip(lab_u8[..., 2], 0, 255)
    rgb = cv2.cvtColor(lab_u8.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return _ensure_float01(rgb)

def _rgb_to_ycrcb(rgb01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(_to_uint8(rgb01), cv2.COLOR_RGB2YCrCb).astype(np.float32)

def _ycrcb_to_rgb(ycrcb: np.ndarray) -> np.ndarray:
    ycrcb_u8 = np.clip(ycrcb, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(ycrcb_u8, cv2.COLOR_YCrCb2RGB)
    return _ensure_float01(rgb)

def _auto_contrast_limits(img01: np.ndarray, p_low: float, p_high: float) -> Tuple[float, float]:
    """Percentile-based limits in [0,1]."""
    lo = float(np.percentile(img01, p_low))
    hi = float(np.percentile(img01, p_high))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi

def _rescale_intensity(img01: np.ndarray, in_low: float, in_high: float) -> np.ndarray:
    """Rescale [in_low, in_high] -> [0,1] with clipping."""
    out = (img01 - in_low) / max(in_high - in_low, 1e-6)
    return np.clip(out, 0.0, 1.0)

def _plot_histogram(img01: np.ndarray, title: str, is_gray: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.set_title(title)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")

    if is_gray:
        ax.hist(img01.ravel(), bins=256, range=(0, 1), alpha=0.9)
    else:
        # For color, show per-channel histograms of RGB
        if img01.dtype != np.float32 and img01.dtype != np.float64:
            img01 = _ensure_float01(img01)
        R, G, B = img01[..., 0].ravel(), img01[..., 1].ravel(), img01[..., 2].ravel()
        ax.hist(R, bins=256, range=(0, 1), alpha=0.6, label="R")
        ax.hist(G, bins=256, range=(0, 1), alpha=0.6, label="G")
        ax.hist(B, bins=256, range=(0, 1), alpha=0.6, label="B")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig

def _download_bytes(img01: np.ndarray, filename: str) -> Tuple[bytes, str]:
    """Convert [0,1] float image to PNG bytes."""
    out = _to_uint8(img01)
    pil = Image.fromarray(out if out.ndim == 2 else out, mode="L" if out.ndim == 2 else "RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue(), filename

# ===========================
# ===== DEMO IMAGE ==========
# ===========================
def _demo_image(color: bool = True) -> np.ndarray:
    """Generate a clean demo image (no internet dependency)."""
    H, W = 360, 540
    if color:
        base = np.zeros((H, W, 3), dtype=np.float32)
        # gradient background
        gx = np.linspace(0, 1, W, dtype=np.float32)
        gy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        base[..., 0] = gx  # R
        base[..., 1] = gy  # G
        base[..., 2] = 0.4 + 0.4 * (np.sin(4 * math.pi * gx)[None, :])  # B
        # add shapes
        cv2.circle(base, (W//3, H//2), 60, (1.0, 0.2, 0.2), -1)
        cv2.rectangle(base, (W//2, H//3), (W-40, 2*H//3), (0.2, 0.9, 0.2), -1)
        return np.clip(base, 0, 1)
    else:
        img = np.zeros((H, W), dtype=np.float32)
        gx = np.linspace(0, 1, W, dtype=np.float32)
        img += gx[None, :]
        cv2.circle(img, (W//2, H//2), 70, 0.15, -1)
        cv2.putText(img, "Demo", (W//6, int(0.8*H)), cv2.FONT_HERSHEY_SIMPLEX, 2, 0.85, 3, cv2.LINE_AA)
        return np.clip(img, 0, 1)

# ==================================
# ===== PROCESSING METHODS ==========
# ==================================

def linear_negative(img01: np.ndarray) -> np.ndarray:
    return 1.0 - img01

def contrast_stretching(img01: np.ndarray, use_percentiles: bool,
                        p_low: float, p_high: float,
                        in_low: float, in_high: float) -> np.ndarray:
    if use_percentiles:
        lo, hi = _auto_contrast_limits(img01, p_low, p_high)
    else:
        lo, hi = in_low, in_high
    return _rescale_intensity(img01, lo, hi)

def piecewise_linear(img01: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Simple 3-piece linear mapping defined by (0,0)->(x1,y1)->(x2,y2)->(1,1)."""
    x1, x2 = np.clip(x1, 0, 1), np.clip(x2, 0, 1)
    y1, y2 = np.clip(y1, 0, 1), np.clip(y2, 0, 1)
    if x2 <= x1:
        x2 = min(1.0, x1 + 1e-6)

    out = np.empty_like(img01)
    # Segment 1: [0, x1]
    m1 = y1 / max(x1, 1e-6) if x1 > 0 else 0.0
    b1 = 0.0
    # Segment 2: (x1, x2]
    m2 = (y2 - y1) / max(x2 - x1, 1e-6)
    b2 = y1 - m2 * x1
    # Segment 3: (x2, 1]
    m3 = (1.0 - y2) / max(1.0 - x2, 1e-6)
    b3 = y2 - m3 * x2

    out = np.where(
        img01 <= x1, m1 * img01 + b1,
        np.where(img01 <= x2, m2 * img01 + b2, m3 * img01 + b3)
    )
    return np.clip(out, 0.0, 1.0)

def log_transform(img01: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """s = gain * log(1 + img) normalized to [0,1]."""
    s = gain * np.log1p(img01)
    s /= (gain * np.log1p(1.0) + 1e-6)
    return np.clip(s, 0.0, 1.0)

def gamma_transform(img01: np.ndarray, gamma: float = 1.0, gain: float = 1.0) -> np.ndarray:
    out = gain * (img01 ** max(gamma, 1e-6))
    return np.clip(out, 0.0, 1.0)

def hist_equalization_gray(img01_gray: np.ndarray) -> np.ndarray:
    u8 = _to_uint8(img01_gray)
    eq = cv2.equalizeHist(u8)
    return _ensure_float01(eq)

def hist_equalization_color_rgb(img01_rgb: np.ndarray) -> np.ndarray:
    """Equalize luminance (Y) in YCrCb to avoid color shifting."""
    ycc = _rgb_to_ycrcb(img01_rgb)
    y = ycc[..., 0].astype(np.uint8)
    y_eq = cv2.equalizeHist(y)
    ycc[..., 0] = y_eq.astype(np.float32)
    return _ycrcb_to_rgb(ycc)

def adaptive_hist_equalization_gray(img01_gray: np.ndarray, kernel_size: int = 8, nbins: int = 256) -> np.ndarray:
    # Use skimage-like approach via CLAHE with high clip limit to mimic AHE behavior (clipLimit very high)
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(max(2, kernel_size), max(2, kernel_size)))
    out = clahe.apply(_to_uint8(img01_gray))
    return _ensure_float01(out)

def adaptive_hist_equalization_color(img01_rgb: np.ndarray, kernel_size: int = 8) -> np.ndarray:
    lab = _rgb_to_lab(img01_rgb)
    L = lab[..., 0].astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(max(2, kernel_size), max(2, kernel_size)))
    L_eq = clahe.apply(L)
    lab[..., 0] = L_eq.astype(np.float32)
    return _lab_to_rgb(lab)

def clahe_gray(img01_gray: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=max(0.01, clip_limit), tileGridSize=(max(2, tile), max(2, tile)))
    out = clahe.apply(_to_uint8(img01_gray))
    return _ensure_float01(out)

def clahe_color(img01_rgb: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    lab = _rgb_to_lab(img01_rgb)
    L = lab[..., 0].astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=max(0.01, clip_limit), tileGridSize=(max(2, tile), max(2, tile)))
    Lc = clahe.apply(L)
    lab[..., 0] = Lc.astype(np.float32)
    return _lab_to_rgb(lab)

# ==================================
# ======== SIDEBAR INPUTS ==========
# ==================================
with st.sidebar:
    st.header("1) Upload Image")
    file = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    mode = st.radio("Image Type", ["Auto-detect", "Grayscale", "Color"], index=0, help="Force processing as grayscale or color.")
    st.divider()
    st.header("2) Choose Method")
    method = st.selectbox(
        "Processing Method",
        [
            "Linear Negative",
            "Contrast Stretching",
            "Piecewise Linear Transformation",
            "Log Transformation",
            "Gamma Transformation",
            "Histogram Equalization",
            "Adaptive Histogram Equalization",
            "CLAHE (Contrast Limited AHE)"
        ],
        index=1
    )

# ==================================
# ======== LOAD / PREP IMAGE ========
# ==================================
if file is not None:
    pil = Image.open(file).convert("RGB")  # keep RGB; convert to gray later if needed
    img_rgb01 = _ensure_float01(np.array(pil))
else:
    st.info("No image uploaded — using a built-in demo image. Upload your own to test on real data.")
    img_rgb01 = _demo_image(color=True)

# Detect or enforce grayscale vs color
if mode == "Grayscale":
    is_gray = True
elif mode == "Color":
    is_gray = False
else:
    is_gray = False  # by default treat uploaded image as color
    # If user’s demo or uploaded is truly grayscale (all channels equal), detect that:
    if np.allclose(img_rgb01[..., 0], img_rgb01[..., 1]) and np.allclose(img_rgb01[..., 1], img_rgb01[..., 2]):
        is_gray = True

img_input = img_rgb01[..., 0] if is_gray else img_rgb01  # grayscale = take R (same as G,B if truly gray)

# ==================================
# ======== METHOD CONTROLS ==========
# ==================================
with st.sidebar:
    st.subheader("3) Parameters")
    if method == "Contrast Stretching":
        use_p = st.checkbox("Use percentiles", value=True)
        if use_p:
            p_lo = st.slider("Low percentile", 0.0, 20.0, 2.0, 0.1)
            p_hi = st.slider("High percentile", 80.0, 100.0, 98.0, 0.1)
            in_lo, in_hi = 0.0, 1.0
        else:
            in_lo = st.slider("Input low", 0.0, 1.0, 0.0, 0.01)
            in_hi = st.slider("Input high", 0.0, 1.0, 1.0, 0.01)
            p_lo, p_hi = 2.0, 98.0
    elif method == "Piecewise Linear Transformation":
        x1 = st.slider("x1 (breakpoint 1)", 0.0, 1.0, 0.25, 0.01)
        y1 = st.slider("y1 (output at x1)", 0.0, 1.0, 0.10, 0.01)
        x2 = st.slider("x2 (breakpoint 2)", 0.0, 1.0, 0.75, 0.01)
        y2 = st.slider("y2 (output at x2)", 0.0, 1.0, 0.90, 0.01)
    elif method == "Log Transformation":
        gain = st.slider("Gain", 0.1, 5.0, 1.0, 0.1)
    elif method == "Gamma Transformation":
        gamma = st.slider("Gamma (γ)", 0.1, 5.0, 1.8, 0.1)
        gain_g = st.slider("Gain", 0.1, 5.0, 1.0, 0.1)
    elif method == "Adaptive Histogram Equalization":
        ksize = st.slider("Kernel / Tile size", 2, 64, 8, 1)
    elif method == "CLAHE (Contrast Limited AHE)":
        clip = st.slider("Clip Limit", 0.1, 20.0, 2.0, 0.1)
        tile = st.slider("Tile Grid Size", 2, 64, 8, 1)

# ==================================
# ======== APPLY PROCESSING =========
# ==================================
def apply_method(img, is_gray_flag: bool):
    if method == "Linear Negative":
        return linear_negative(_ensure_float01(img))

    if method == "Contrast Stretching":
        if is_gray_flag:
            return contrast_stretching(_ensure_float01(img), use_p, p_lo, p_hi, in_lo, in_hi)
        else:
            chs = []
            for c in range(3):
                ch = img[..., c]
                chs.append(contrast_stretching(_ensure_float01(ch), use_p, p_lo, p_hi, in_lo, in_hi))
            return np.dstack(chs)

    if method == "Piecewise Linear Transformation":
        if is_gray_flag:
            return piecewise_linear(_ensure_float01(img), x1, y1, x2, y2)
        else:
            return np.dstack([piecewise_linear(_ensure_float01(img[..., c]), x1, y1, x2, y2) for c in range(3)])

    if method == "Log Transformation":
        if is_gray_flag:
            return log_transform(_ensure_float01(img), gain=gain)
        else:
            return np.dstack([log_transform(_ensure_float01(img[..., c]), gain=gain) for c in range(3)])

    if method == "Gamma Transformation":
        if is_gray_flag:
            return gamma_transform(_ensure_float01(img), gamma=gamma, gain=gain_g)
        else:
            return np.dstack([gamma_transform(_ensure_float01(img[..., c]), gamma=gamma, gain=gain_g) for c in range(3)])

    if method == "Histogram Equalization":
        if is_gray_flag:
            return hist_equalization_gray(_ensure_float01(img))
        else:
            return hist_equalization_color_rgb(_ensure_float01(img))

    if method == "Adaptive Histogram Equalization":
        if is_gray_flag:
            return adaptive_hist_equalization_gray(_ensure_float01(img), kernel_size=ksize)
        else:
            return adaptive_hist_equalization_color(_ensure_float01(img), kernel_size=ksize)

    if method == "CLAHE (Contrast Limited AHE)":
        if is_gray_flag:
            return clahe_gray(_ensure_float01(img), clip_limit=clip, tile=tile)
        else:
            return clahe_color(_ensure_float01(img), clip_limit=clip, tile=tile)

    return _ensure_float01(img)

img_before = _ensure_float01(img_input)
img_after = apply_method(img_before, is_gray)

# ==================================
# ======== DISPLAY RESULTS =========
# ==================================
colA, colB = st.columns(2, gap="large")
with colA:
    st.subheader("Before")
    st.image(_to_uint8(img_before), caption="Original", use_container_width=True, channels="GRAY" if is_gray else "RGB")
    fig_b = _plot_histogram(img_before, "Histogram (Before)", is_gray)
    st.pyplot(fig_b, use_container_width=True)

with colB:
    st.subheader("After")
    st.image(_to_uint8(img_after), caption=f"Processed — {method}", use_container_width=True, channels="GRAY" if is_gray else "RGB")
    fig_a = _plot_histogram(img_after, "Histogram (After)", is_gray)
    st.pyplot(fig_a, use_container_width=True)

# ==================================
# ========== DOWNLOADS =============
# ==================================
st.divider()
c1, c2 = st.columns([1, 2])
with c1:
    out_bytes, fname = _download_bytes(img_after, "processed.png")
    st.download_button("⬇️ Download Processed PNG", data=out_bytes, file_name=fname, mime="image/png")
with c2:
    st.markdown(
        f"**Method:** `{method}` &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**Mode:** `{'Grayscale' if is_gray else 'Color'}`"
    )

# ==================================
# ======= PERFORMANCE NOTES ========
# ==================================
st.caption(
    "⚙️ Notes: For color, histogram equalization & AHE/CLAHE operate on luminance (Y/L) "
    "to preserve color fidelity. All methods operate in float [0,1] for numerical stability."
)
