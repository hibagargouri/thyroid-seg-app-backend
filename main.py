# =============================================================
# Backend for the thyroid detection app — RENDER VERSION
#
# Only the segmentation model now (generation was dropped from the
# app) — much lighter than our earlier attempt, which had both
# models loaded and hit Render's 512MB free-tier memory limit.
# This one should fit comfortably.
# =============================================================

import os
import io
import base64
import urllib.request
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

DEVICE = "cpu"  # Render's free tier is CPU-only
torch.set_num_threads(1)

RESOLUTION = 256
SEG_CKPT = "thyroid_seg_unet.pt"
SEG_CKPT_URL = "https://www.dropbox.com/scl/fi/t8z3tlams0pa7ztptm1h0/thyroid_seg_unet.pt.zip?rlkey=ovb2bd9ab7wiih4bb1qyy9xhg&st=c8nmvkoh&dl=1"

def ensure_checkpoint(local_path, url):
    if not os.path.isfile(local_path):
        print(f"Downloading {local_path} ...")
        urllib.request.urlretrieve(url, local_path)
        print(f"Downloaded {local_path} ({os.path.getsize(local_path)/1e6:.1f} MB)")

ensure_checkpoint(SEG_CKPT, SEG_CKPT_URL)


# ---------------------------------------------------------------
# Model architecture — identical to training
# ---------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class SegUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 8)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


# ---------------------------------------------------------------
# Load model once at startup — just this one small model this time
# ---------------------------------------------------------------
print("Loading segmentation model...")
seg_model = SegUNet().to(DEVICE)
seg_model.load_state_dict(torch.load(SEG_CKPT, map_location=DEVICE))
seg_model.eval()
print("Model loaded — ready.")


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def preprocess(pil_img, resolution=RESOLUTION):
    img = pil_img.convert("L")
    w, h = img.size
    scale = resolution / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - resolution) // 2
    top = (new_h - resolution) // 2
    return img.crop((left, top, left + resolution, top + resolution))

def to_tensor_image(pil_img):
    arr = (np.asarray(pil_img).astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------
# API
# ---------------------------------------------------------------
app = FastAPI(title="Thyroid Detection API")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Thyroid detection API is running on Render."}

@app.post("/detect_mask")
def detect_mask(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    pil_img = Image.open(io.BytesIO(image_bytes))
    img = preprocess(pil_img)
    x = to_tensor_image(img).to(DEVICE)
    with torch.no_grad():
        pred = torch.sigmoid(seg_model(x))
    mask_arr = (pred[0, 0].numpy() > 0.5).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask_arr, mode="L")
    return JSONResponse({
        "preprocessed_image_base64": pil_to_base64(img),
        "mask_base64": pil_to_base64(mask_pil),
    })
