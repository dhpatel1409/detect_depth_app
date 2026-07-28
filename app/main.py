import cv2
import matplotlib.pyplot as plt
import os
img = os.path.join(os.path.dirname(__file__),"..","sample_images","00555_colors.png")
img = cv2.imread(img)


from pipeline import run_pipeline
annotated, detections = run_pipeline(img)
write_path = os.path.join(os.path.dirname(__file__),"..","output","00555_colors_annotated.png")
cv2.imwrite(write_path, annotated)   
# plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
# plt.show()