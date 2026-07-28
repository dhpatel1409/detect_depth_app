import torch
import time
import sys
import os

# Make the vendored Depth-Anything-V2 code importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "vendor", "Depth-Anything-V2", "metric_depth"))
from depth_anything_v2.dpt import DepthAnythingV2
from huggingface_hub import hf_hub_download

_model = None

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

def load_model():
    global _model
    if _model is None:
        checkpoint_path = hf_hub_download(
            repo_id="dhpatel1409/depth-anything-nyud-finetuned",  # update once you've created this repo
            filename="DepthAnythingV2 NYUd2.pth"
        )
        _model = DepthAnythingV2(**{**MODEL_CONFIGS['vits'], 'max_depth': 10})
        _model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        _model.eval()
    return _model

def depth_pred(raw_img):
    model = load_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # raw_img = torch.from_numpy(raw_img).permute(2, 0, 1).unsqueeze(0).float().to('cpu')
    depth = model.infer_image(raw_img)  # HxW depth map in meters, numpy
    if torch.is_tensor(depth):
        depth = depth.cpu().numpy()
    return depth