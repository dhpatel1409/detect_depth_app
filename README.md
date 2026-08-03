# Depth-Aware Object Detection

Detects objects in an image and estimates their real-world distance — in meters — by fusing a fine-tuned monocular depth model with an object detector. No stereo camera, no LiDAR, just a single RGB image.

**🔗 Live demo:** https://depth-detect-app-8gpfbvwnb3skdvbepxmobp.streamlit.app/
**🖼️ Frontend repo:** [detect-depth-streamlit](https://github.com/dhpatel1409/depth-detect-streamlit.git)

---

## What it does

Most object detectors tell you *what's* in an image. This pipeline also tells you *how far away* each object is:

1. **YOLOv8** detects objects and draws bounding boxes
2. **Depth-Anything V2** (fine-tuned on NYUDv2) predicts a per-pixel depth map in meters
3. **Fusion logic** samples the depth map inside each bounding box (median, for robustness to box edges/occlusion) to estimate that object's distance
4. Output: the original image annotated with class, confidence, and distance for every detected object

## Example
 
| Input | Output |
|---|---|
| ![Input](sample_images/00555_colors.png) | ![output](output/00555_colors_annotated.png) |
 
*Raw image (left) vs. detected objects annotated with class, confidence, and estimated distance in meters (right).*
## Architecture

```
Input Image
    │
    ├──────────────▶ YOLOv8 (COCO-pretrained) ──▶ bounding boxes + classes
    │
    └──────────────▶ Depth-Anything V2 (fine-tuned) ──▶ per-pixel depth map (meters)
                              │
                              ▼
                    Fusion: median depth per box
                              │
                              ▼
                  Annotated image + distance per object
```

Served via a **FastAPI** backend, deployed on **Modal** (serverless, GPU-accelerated).

## Tech stack

`Python` · `PyTorch` · `OpenCV` · `Ultralytics YOLOv8` · `Depth-Anything V2` · `FastAPI` · `Docker` · `Modal` · `Hugging Face Hub` . `Streamlit` . `Cloud deployment`

## Performance

| Deployment | Hardware | Latency / image |
|---|---|---|
| Modal | CPU only | ~6s |
| Modal | NVIDIA T4 (GPU) | **~700ms** |

Deployed on GPU for the live demo.

## Project structure

```
├── app/
│   ├── depth_model.py     # loads fine-tuned Depth-Anything V2, runs inference
│   ├── detector.py        # YOLOv8 object detection
│   ├── fusion.py          # fuses detections with depth map, draws annotations
│   ├── pipeline.py         # end-to-end run_pipeline()
│   ├── eval_utils.py       # depth evaluation metrics (AbsRel, RMSE, δ1-3)
│   └── main.py             # FastAPI app (/health, /predict, /predict-annotated)
├── vendor/
│   └── Depth-Anything-V2/  # upstream model code
├── modal_app.py            # Modal deployment entrypoint
├── Dockerfile
├── requirements.txt
└── weights/                 # gitignored — pulled from Hugging Face Hub at runtime
```

## Model

- **Detection:** YOLOv8n, pretrained on COCO — no fine-tuning
- **Depth:** Depth-Anything V2 (ViT-S, metric depth head), fine-tuned on **NYUDv2** (indoor scenes). Outputs true metric depth in meters, not relative depth ordering.
- **Weights:** hosted on [Hugging Face Hub](https://huggingface.co), downloaded at container startup — kept out of the Docker image and out of Git.

## Running locally

```bash
git clone https://github.com/dhpatel1409/detect_depth_app
cd detect_depth_app
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Visit `http://localhost:8000/docs` for the interactive API (Swagger UI).

## Running with Docker

```bash
docker build -t detect-depth-app .
docker run -p 8000:8000 detect-depth-app
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/predict` | POST | Returns detections + distances as JSON |
| `/predict-image` | POST | Returns the annotated image (JPEG) |

## Deployment journey

Getting this running on a free tier took a few detours worth documenting:

- **Render** — fit and ran, but 512MB free-tier RAM couldn't hold two vision models simultaneously (OOM crash under load)
- **Hugging Face Spaces** — Docker SDK now requires a paid plan on personal accounts
- **Google Cloud Run** — card verification in India required a large prepayment hold
- **Modal** — landed here: pay-per-second billing, configurable memory, and on-demand GPU access solved both the memory ceiling and the latency problem in one move

## Evaluation

- ✅ Depth metrics on NYUDv2 test set (AbsRel, RMSE, δ1/δ2/δ3) — implemented in `eval_utils.py`
- 🔲 YOLO detection metrics (mAP, precision, recall) via a manually annotated NYUDv2 subset (Roboflow)

## Roadmap

- [ ] ONNX export + quantization for lower latency
- [ ] Real-time video inference
- [ ] Manual annotation + YOLO evaluation (mAP/precision/recall) on NYUDv2 subset


## Author

**Dharmik Patel** — M.Tech, Vision & Intelligent Systems, IIT Kharagpur
[LinkedIn](https://www.linkedin.com/in/dharmik-patel-a88b94280) · [GitHub](https://github.com/dhpatel1409)
