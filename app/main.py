import cv2
import matplotlib.pyplot as plt
img = cv2.imread(r"D:\project\detect + depth\depth - detect repo\sample_images\00555_colors.png")

from pipeline import run_pipeline
annotated, detections = run_pipeline(img)
cv2.imwrite(r"D:\project\detect + depth\depth - detect repo\output\00555_colors_annotated.png", annotated)   
# plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
# plt.show()