import time
from app.depth_model import depth_pred
from app.detector import detect_objects
from app.fusion import fuse_detections_with_depth, draw_detections

def run_pipeline(raw_img):

    start_time = time.time()
    pred_depth = depth_pred(raw_img)
    end_time = time.time()
    # print(f"Depth prediction time: {end_time - start_time} seconds")
    start_time = time.time()
    detections = detect_objects(raw_img)
    end_time = time.time()
    # print(f"Object detection time: {end_time - start_time} seconds")
    start_time = time.time()
    detections = fuse_detections_with_depth(detections, pred_depth)
    end_time = time.time()
    # print(f"Depth fusion time: {end_time - start_time} seconds")
    annotated = draw_detections(raw_img, detections)
    return annotated, detections