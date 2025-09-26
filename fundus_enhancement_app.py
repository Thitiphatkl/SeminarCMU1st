# app.py
# Retinal fundus enhancement from:
# Wang, Li, Yang, "Retinal fundus image enhancement with image decomposition and visual adaptation"
# Computers in Biology and Medicine 128 (2021) 104116

import io
import math
from typing import Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ------------------------------
# Safe OpenCV import hint
# ------------------------------
# If cv2 not found: pip install opencv-python-headless

st.set_page_config(page_title="Retinal Fundus Enhancement", page_icon="🩺", layout="wide")
st.title("🩺 Retinal Fundus Enhancement (Decomposition + Visual Adaptation)")

# ------------------------------
# Utilities
# ------------------------------
def to_float01(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    img = img.astype(np.float32)
    m, M = float(img.min()), float(img.max())
    if M <= m:
        return np.zeros_like(img, dtype=np.float32)
    return (img - m) / (M - m)

def to_uint8(img01):
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255.0 + 0.5).astype(np.uint8)

def rgb_to_hsv(rgb01):
    return cv2.cvtColor(to_uint8(rgb01), cv2.COLOR_RGB2HSV).astype(np.float32)

def hsv_to_rgb(hsv):
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return to_float01(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))

def gaussian_from_sigma(sigma: float):
    # map sigma -> odd kernel size
    k = max(3, int(6 * sigma + 1))
    if k % 2 == 0:
        k += 1
    return (k, k), sigma

def demo_fundus():
    # Synthetic demo with green-ish disc and vessels-like strokes
    H, W = 640, 640
    img = np.zeros((H, W, 3), np.float32)
    rr, cc = np.ogrid[:H, :W]
    cx, cy, r = W//2, H//2, int(0.8*H/2)
    mask = (rr-cy)**2 + (cc-cx)**2 <= r*r
    img[...,1][mask] = 0.15 + 0.5*(1 - ((rr[mask]-cy)**2 + (cc[mask]-cx)**2)/float(r*r))
    cv2.circle(img, (cx-80, cy-60), 42, (0.3, 0.25, 0.2), -1)
    for x in range(-240, 241, 40):
        cv2.line(img, (cx+x, cy-220), (cx+x//2, cy+220), (0.1, 0.5, 0.1), 1)
    return np.clip(img, 0, 1)

# ------------------------------
# Chambolle ROF TV denoising (grayscale), small, fast implementation
# ------------------------------
def tv_denoise_gray(u0: np.ndarray, weight: float, tau: float = 0.125, iters: int = 50) -> np.ndarray:
    # u0 in [0,1], float32; returns base (structure)
    u = u0.copy()
    px = np.zeros_like(u0)
    py = np.zeros_like(u0)
    lam = weight  # lambda in ROF
    for _ in range(iters):
        # gradient of u
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u
        # update dual
        px_new = px + (tau/lam) * ux
        py_new = py + (tau/lam) * uy
        norm = np.maximum(1.0, np.sqrt(px_new*px_new + py_new*py_new))
        px, py = px_new / norm, py_new / norm
        # divergence
        div = (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))
        # primal
        u = (u0 + lam * div)
    return np.clip(u, 0.0, 1.0)

def tv_decompose_color(img01: np.ndarray, lam: Tuple[float, float, float], iters: int = 50):
    # per-channel TV denoise -> base; detail = structure - base
    base = np.zeros_like(img01)
    for c in range(3):
        base[..., c] = tv_denoise_gray(img01[..., c], lam[c], iters=iters)
    detail = img01 - base
    return base, detail

# ------------------------------
# Noise estimation (λ1) and kernel Ns (Immerkaer)
# ------------------------------
NS = np.array([[1, -2, 1],
               [-2, 4, -2],
               [1, -2, 1]], np.float32)

def lambda1_global_noise(img01: np.ndarray) -> Tuple[float, float, float]:
    H, W = img01.shape[:2]
    lams = []
    for c in range(3):
        ch = img01[..., c]
        conv = cv2.filter2D(ch, -1, NS, borderType=cv2.BORDER_REFLECT)
        s = np.sum(np.abs(conv))
        lam = math.sqrt(math.pi/2.0) / (6.0 * (W-2) * (H-2)) * s
        lams.append(lam + 1e-6)  # avoid 0
    return tuple(lams)

# ------------------------------
# Visual adaptation on base (Naka–Rushton)
# ------------------------------
def visual_adaptation_on_base(base_rgb01: np.ndarray, n: float = 1.0):
    hsv = rgb_to_hsv(base_rgb01)
    V = to_float01(hsv[..., 2])
    Mg = float(np.mean(V))
    Sg = float(np.std(V))
    sigma_g = Mg / (1.0 + math.exp(Sg))
    # Naka–Rushton
    outV = (V**n) / ( (V**n) + (sigma_g**n + 1e-12) )
    hsv[..., 2] = np.clip(outV*255.0, 0, 255)
    return hsv_to_rgb(hsv)

# ------------------------------
# Weighted fusion (discard noise layer)
# ------------------------------
def fuse(enh_base_rgb01: np.ndarray, detail_rgb01: np.ndarray,
         alpha_r: float, alpha_g: float, alpha_b: float, gauss_sigma: float):
    # spatial weights from blurred |detail|
    mag = np.abs(detail_rgb01)
    (kx, ky), sig = gaussian_from_sigma(gauss_sigma)
    w = np.zeros_like(detail_rgb01)
    for c, a in enumerate([alpha_r, alpha_g, alpha_b]):
        blur = cv2.GaussianBlur(mag[..., c], (kx, ky), sig, borderType=cv2.BORDER_REFLECT)
        w[..., c] = a * blur
    return np.clip(enh_base_rgb01 + w * detail_rgb01, 0.0, 1.0)

# ------------------------------
# Sidebar — Upload & Parameters
# ------------------------------
with st.sidebar:
    st.header("1) Upload fundus image")
    f = st.file_uploader("PNG/JPG", type=["png", "jpg", "jpeg"])
    st.caption("If none, a synthetic demo fundus is used.")

    st.header("2) Defaults")
    use_paper_defaults = st.checkbox("Use paper defaults", True,
                                     help="λ₂=0.3, α_R=α_G=600, α_B=0, Gaussian σ=10, n=1.0")

    st.header("3) Decomposition")
    iters = st.slider("TV iterations", 20, 200, 60, 5)
    lam2 = 0.3 if use_paper_defaults else st.slider("λ₂ (2nd TV)", 0.05, 1.0, 0.3, 0.01)

    st.header("4) Visual adaptation")
    n_naka = 1.0 if use_paper_defaults else st.slider("Naka–Rushton n", 0.5, 3.0, 1.0, 0.1)

    st.header("5) Fusion weights")
    alpha_r = 600.0 if use_paper_defaults else st.number_input("α_R (veins)", 0.0, 2000.0, 600.0, 10.0)
    alpha_g = 600.0 if use_paper_defaults else st.number_input("α_G (arteries)", 0.0, 2000.0, 600.0, 10.0)
    alpha_b = 0.0 if use_paper_defaults else st.number_input("α_B (blue/artifacts)", 0.0, 2000.0, 0.0, 10.0)
    gauss_sigma = 10.0 if use_paper_defaults else st.slider("Gaussian σ (for ω)", 1.0, 20.0, 10.0, 0.5)

# ------------------------------
# Load image
# ------------------------------
if f is not None:
    img = Image.open(f).convert("RGB")
    rgb01 = to_float01(np.array(img))
else:
    rgb01 = demo_fundus()

st.caption("Pipeline: (1) TV decomposition for noise & structure → (2) TV for base/detail → (3) luminance adaptation on base → (4) fuse enhanced base + weighted detail; discard noise.")

# ------------------------------
# STEP 1: noise-vs-structure (λ1 from global noise estimate)
# ------------------------------
lam1_r, lam1_g, lam1_b = lambda1_global_noise(rgb01)
structure, noise = tv_decompose_color(rgb01, (lam1_r, lam1_g, lam1_b), iters=iters)

# ------------------------------
# STEP 2: base/detail from structure (λ₂)
# ------------------------------
base, detail = tv_decompose_color(structure, (lam2, lam2, lam2), iters=iters)

# ------------------------------
# STEP 3: visual adaptation on base (HSV.V, Naka–Rushton)
# ------------------------------
enh_base = visual_adaptation_on_base(base, n=n_naka)

# ------------------------------
# STEP 4: fusion (discard noise layer)
# ------------------------------
out = fuse(enh_base, detail, alpha_r, alpha_g, alpha_b, gauss_sigma)

# ------------------------------
# Display
# ------------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Original")
    st.image(to_uint8(rgb01), channels="RGB", use_container_width=True)
with c2:
    st.subheader("Enhanced")
    st.image(to_uint8(out), channels="RGB", use_container_width=True)

st.divider()
# Optional inspection views
with st.expander("Show intermediate layers"):
    cA, cB, cC = st.columns(3)
    with cA:
        st.caption("Structure (after λ₁ TV)")
        st.image(to_uint8(structure), channels="RGB", use_container_width=True)
    with cB:
        st.caption("Base (after λ₂ TV)")
        st.image(to_uint8(base), channels="RGB", use_container_width=True)
    with cC:
        st.caption("Detail")
        # boost for visibility
        st.image(to_uint8(np.clip(0.5 + detail, 0, 1)), channels="RGB", use_container_width=True)

# Download
buf = io.BytesIO()
Image.fromarray(to_uint8(out)).save(buf, format="PNG")
st.download_button("⬇️ Download enhanced PNG", data=buf.getvalue(), file_name="fundus_enhanced.png", mime="image/png")
