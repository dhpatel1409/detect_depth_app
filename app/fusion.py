import numpy as np
import cv2

def fuse_detections_with_depth(detections, depth_map):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(depth_map.shape[1], x2), min(depth_map.shape[0], y2)
        depth_crop = depth_map[y1:y2, x1:x2]
        if depth_crop.size == 0:
            det["distance_m"] = None
            continue
        det["distance_m"] = float(np.median(depth_crop))
    return detections

def draw_detections(raw_img, detections):
    img = raw_img.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        dist = det["distance_m"]
        label = f'{det["class"]} {det["confidence"]:.2f} | {dist:.2f}m' if dist else det["class"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img