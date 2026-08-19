# =============================================================
# Backend for the thyroid data-augmentation app — FastAPI version.
# Deploy this on Render.com (free tier).
#
# Endpoints:
#   POST /detect_mask  -> upload an image, get back the detected mask
#   POST /generate      -> upload a mask + num_variants,
#                           get back N synthetic variants
#
# Images are sent/received as base64-encoded PNGs in JSON — the
# standard way for a mobile app to talk to an API like this.
# =============================================================

import os
import urllib.request
import math
import io
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from diffusers import DDPMScheduler

DEVICE = "cpu"  # Render's free tier is CPU-only
RESOLUTION = 256
INFERENCE_STEPS = 60  # reduced from 1000 for response time on free CPU hardware

SEG_CKPT = "thyroid_seg_unet.pt"
GEN_CKPT = "thyroid_ddpm_conditional.pt"

# Hosted on Dropbox (dl=1 forces direct download instead of a preview page)
SEG_CKPT_URL = "https://www.dropbox.com/scl/fi/t8z3tlams0pa7ztptm1h0/thyroid_seg_unet.pt.zip?rlkey=ovb2bd9ab7wiih4bb1qyy9xhg&st=c8nmvkoh&dl=1"
GEN_CKPT_URL = "https://www.dropbox.com/scl/fi/1qzx98grw8jet54t67efw/thyroid_ddpm_conditional.pt.zip?rlkey=9ozxs3c3hqyosjtkzq7nx7tsx&st=y0u7198q&dl=1"

def ensure_checkpoint(local_path, url):
    if not os.path.isfile(local_path):
        print(f"Downloading {local_path} from {url} ...")
        urllib.request.urlretrieve(url, local_path)
        print(f"Downloaded {local_path} ({os.path.getsize(local_path)/1e6:.1f} MB)")

ensure_checkpoint(SEG_CKPT, SEG_CKPT_URL)
ensure_checkpoint(GEN_CKPT, GEN_CKPT_URL)


# ---------------------------------------------------------------
# Model architectures — identical to training (must match exactly)
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

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.out = nn.Conv2d(ch, ch, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, c, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)
        out = (v @ attn.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.out(out)

class GenUNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, base_ch=64, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.down1 = ResBlock(base_ch, base_ch, time_dim)
        self.pool1 = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)
        self.down2 = ResBlock(base_ch * 2, base_ch * 2, time_dim)
        self.pool2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)
        self.down3 = ResBlock(base_ch * 4, base_ch * 4, time_dim)
        self.pool3 = nn.Conv2d(base_ch * 4, base_ch * 4, 4, stride=2, padding=1)
        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, time_dim)
        self.mid_attn = SelfAttention(base_ch * 4)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, time_dim)
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 4, stride=2, padding=1)
        self.upres3 = ResBlock(base_ch * 8, base_ch * 4, time_dim)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.upres2 = ResBlock(base_ch * 4, base_ch * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.upres1 = ResBlock(base_ch * 2, base_ch, time_dim)
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_ch, 3, padding=1)
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h0 = self.in_conv(x)
        h1 = self.down1(h0, t_emb)
        h2 = self.down2(self.pool1(h1), t_emb)
        h3 = self.down3(self.pool2(h2), t_emb)
        m = self.mid1(self.pool3(h3), t_emb)
        m = self.mid_attn(m)
        m = self.mid2(m, t_emb)
        u3 = self.upres3(torch.cat([self.up3(m), h3], dim=1), t_emb)
        u2 = self.upres2(torch.cat([self.up2(u3), h2], dim=1), t_emb)
        u1 = self.upres1(torch.cat([self.up1(u2), h1], dim=1), t_emb)
        return self.out_conv(F.silu(self.out_norm(u1)))


# ---------------------------------------------------------------
# Load models once at startup
# ---------------------------------------------------------------
app = FastAPI(title="Thyroid Data Augmentation API")

print("Loading segmentation model...")
seg_model = SegUNet().to(DEVICE)
seg_model.load_state_dict(torch.load(SEG_CKPT, map_location=DEVICE))
seg_model.eval()

print("Loading generation model...")
gen_model = GenUNet(in_ch=2, out_ch=1).to(DEVICE)
gen_model.load_state_dict(torch.load(GEN_CKPT, map_location=DEVICE))
gen_model.eval()

scheduler = DDPMScheduler(
    num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2",
    clip_sample=True, clip_sample_range=1.0, prediction_type="epsilon",
)
print("Models loaded — ready.")


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

def to_tensor_mask(pil_img):
    arr = (np.asarray(pil_img.convert("L")).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
    arr = arr * 2 - 1
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

def tensor_to_base64(t):
    arr = ((t.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()[0, 0]
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def base64_to_pil(b64_str):
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Thyroid data augmentation API is running."}


@app.post("/detect_mask")
async def detect_mask(file: UploadFile = File(...)):
    image_bytes = await file.read()
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


@app.post("/generate")
async def generate(mask_base64: str = Form(...), num_variants: int = Form(1)):
    mask_img = base64_to_pil(mask_base64).convert("L")
    if mask_img.size != (RESOLUTION, RESOLUTION):
        mask_img = preprocess(mask_img)
    mask_t = to_tensor_mask(mask_img).to(DEVICE)

    n = max(1, min(int(num_variants), 5))  # cap at 5 to protect free-tier resources
    masks_batch = mask_t.repeat(n, 1, 1, 1)

    scheduler.set_timesteps(INFERENCE_STEPS)
    with torch.no_grad():
        image = torch.randn(n, 1, RESOLUTION, RESOLUTION, device=DEVICE)
        for t in scheduler.timesteps:
            t_batch = torch.full((n,), int(t), device=DEVICE, dtype=torch.long)
            model_input = torch.cat([image, masks_batch], dim=1)
            pred_noise = gen_model(model_input, t_batch)
            image = scheduler.step(pred_noise, t, image).prev_sample
        image = image.clamp(-1, 1)

    results_base64 = [tensor_to_base64(image[i:i+1]) for i in range(n)]
    return JSONResponse({"variants_base64": results_base64})
