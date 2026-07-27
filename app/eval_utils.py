import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def render_depth(values, colormap_name="magma_r"):
    if values is None:
        return None
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)
    colormap = plt.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True)
    colors = colors[:, :, :3]
    return Image.fromarray(colors)

def errors(gt, pred):
    valid_mask = (gt > 0) & (gt < 10)
    pred_eval, gt_eval = pred[valid_mask], gt[valid_mask]
    threshold = np.maximum((gt_eval / pred_eval), (pred_eval / gt_eval))
    delta1 = (threshold < 1.25).mean()
    delta2 = (threshold < 1.25 ** 2).mean()
    delta3 = (threshold < 1.25 ** 3).mean()
    abs_diff = np.abs(pred_eval - gt_eval)
    mae = np.mean(abs_diff)
    rmse = np.sqrt(np.mean(np.power(abs_diff, 2)))
    abs_rel = np.mean(abs_diff / gt_eval)
    log_abs_diff = np.abs(np.log10(pred_eval) - np.log10(gt_eval))
    log_mae = np.mean(log_abs_diff)
    log_rmse = np.sqrt(np.mean(np.power(log_abs_diff, 2)))
    return {"mae": mae, "rmse": rmse, "abs_rel": abs_rel,
            "log_mae": log_mae, "log_rmse": log_rmse,
            "delta1": delta1, "delta2": delta2, "delta3": delta3}