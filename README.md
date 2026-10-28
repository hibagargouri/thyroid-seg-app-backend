# Thyroid Segmentation App — Backend

REST API that segments thyroid nodules in ultrasound images. It is the server behind the Android app built during my 2026 summer internship (ISIMS, supervised by Dr. Adel Alimi and Dr. Yosr Ghozzi).

The app sends an ultrasound image to this API. The API runs a segmentation U-Net and returns the predicted nodule mask together with a *nodule detected / not detected* flag.

> **Research prototype, not a medical device.** It must not be used for diagnosis.

## How it works

1. The uploaded image is converted to grayscale, resized so its short side is 256 px, and center-cropped to 256×256.
2. Pixel values are scaled to [-1, 1] and passed to the U-Net (CPU inference).
3. The sigmoid output is thresholded at 0.5 to give a binary mask.
4. `nodule_detected` is `true` when more than about 0.2% of the mask pixels are white.
5. A simple color check sets `likely_not_ultrasound`: real ultrasounds are grayscale (R ≈ G ≈ B), so an image with strong color differences between channels is flagged. This is a lightweight safety check, not a classifier. It does not recognize grayscale images of other organs.

## Endpoints

### `GET /`

Health check.

```json
{"status": "ok", "message": "Thyroid detection API is running on Render."}
```

### `POST /detect_mask`

Multipart upload with the image in the `file` field.

```bash
curl -X POST "https://<your-service>.onrender.com/detect_mask" \
     -F "file=@ultrasound.png"
```

Response:

```json
{
  "preprocessed_image_base64": "<256x256 grayscale PNG, base64>",
  "mask_base64": "<256x256 binary mask PNG, base64>",
  "likely_not_ultrasound": false,
  "nodule_detected": true
}
```

Interactive documentation is available at `/docs` (FastAPI Swagger UI).

## Model

- Architecture: U-Net (4 encoder levels, bottleneck, 4 decoder levels with skip connections), 1 input channel, 1 output channel, 256×256 input.
- Training data: real TN3K thyroid ultrasound images only (the baseline model, no synthetic data).
- The weights file `thyroid_seg_unet.pt` is downloaded at startup from the URL set in `SEG_CKPT_URL` in `main.py` if it is not already on disk. To use another model, upload it and change that link (and the file name).
- The architecture in `main.py` must stay identical to the one used for training, otherwise the weights will not load.

The training, generation and evaluation code (conditional DDPM, FID, segmentation validation) lives in the companion repository: **thyroid-ultrasound-generation**.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/docs.

`requirements.txt` installs the CPU-only build of PyTorch, which keeps the install small enough for a free hosting tier.

## Deploy on Render

- Environment: Python web service.
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Notes for the free tier:

- The service sleeps after a period without requests, so the first request after a pause can take a while (the model is reloaded).
- Memory is limited to 512 MB. This is why the app serves only the segmentation model.

## Limitations

- Trained on a single public dataset (TN3K); performance on images from other devices is not guaranteed.
- Mask quality varies from image to image.
- The non-ultrasound check is a simple heuristic.
- For research and demonstration only.

## Tech stack

Python · FastAPI · Uvicorn · PyTorch (CPU) · Pillow · NumPy

## Author

Hiba Gargouri — Licence 2nd year, Big Data — Higher Institute of Computer Science and Multimedia of Sfax (ISIMS)
