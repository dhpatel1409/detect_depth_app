from app.depth_model import depth_pred
from app.detector import detect_objects
from app.fusion import fuse_detections_with_depth, draw_detections

def run_pipeline(raw_img):
    pred_depth = depth_pred(raw_img)
    detections = detect_objects(raw_img)
    detections = fuse_detections_with_depth(detections, pred_depth)
    annotated = draw_detections(raw_img, detections)
    return annotated, detections