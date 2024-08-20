# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 18:32:30 2024

@author: h4tec
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Path to the uploaded image
image_path = "C:/Users/h4tec/Desktop/상태평가이미지/상태평가이미지_분류/2. 누수/21_CT23021-02-001_302_1_2_1.jpg"   #C:\Users\h4tec\Desktop\상태평가이미지\상태평가이미지_분류\2. 누수\21_CT23021-02-001_302_4_2_1.jpg



# Load the image using OpenCV
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a threshold to highlight the white text
_, white_thresholded = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)

# Convert the image to HSV color space to identify green and blue text
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define range for green color and threshold
green_lower = np.array([40, 40, 40])
green_upper = np.array([80, 255, 255])
green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

# Define range for blue color and threshold
blue_lower = np.array([100, 40, 40])
blue_upper = np.array([140, 255, 255])
blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

# Combine the masks for white, green, and blue text
combined_mask = cv2.bitwise_or(white_thresholded, green_mask)
combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

# Use inpainting to remove the text
inpainted_image = cv2.inpaint(image, combined_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

# Save the result to display
removed_text_image_path = "/mnt/data/removed_white_green_blue_text_improved.jpg"
cv2.imwrite(removed_text_image_path, inpainted_image)

# Display the result
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
plt.title('Edited Image')
plt.axis('off')

plt.show()

