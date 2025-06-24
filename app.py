from ultralytics import YOLO
import cv2
import os

#Load trained model
model = YOLO(r'C:\Users\smoks\ANIMAL_CLASSIFICATION\best.pt')

#Input image path
image_path = r'C:\Users\smoks\ANIMAL_CLASSIFICATION\Train\images\9efd18dd6c.jpg'

#Run prediction
results = model(image_path, save=True, conf=0.25)

#Get saved image path
output_dir = results[0].save_dir
output_img_path = os.path.join(output_dir, os.path.basename(image_path))

#Load and display result
img = cv2.imread(output_img_path)
cv2.imshow('Detection',img)
cv2.waitKey(0)
cv2.destroyAllWindows()