import cv2
import numpy as np
from roboflow import get_head_locations

# # Load the goose image
# img = cv2.imread('geese/goose1.jpg')

# # Resize the image (optional for easier processing)
# img = cv2.resize(img, (640, 480))

# # Convert the image to HSV color space
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# # Define HSV range for detecting the darkest areas (black/dark brown)
# # Hue 0-180, Saturation 0-255, Value 0-255 (here we focus on dark value areas)
# lower_dark = np.array([0, 0, 0])      # Target very dark regions
# upper_dark = np.array([180, 255, 70])  # You can adjust this for better accuracy

# # Create a mask to isolate the dark regions of the goose
# mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

# # Perform Gaussian blur to reduce noise in the mask
# blurred = cv2.GaussianBlur(mask_dark, (5, 5), 0)

# # Use Canny edge detection on the blurred dark regions
# edges = cv2.Canny(blurred, 50, 150)

# # Find contours in the edge-detected image
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Filter out small or irrelevant contours
# min_contour_area = 300  # Adjust based on the size of dark areas on the goose
# max_contour_area = 5000

# # Draw the contours that are likely to represent the darkest areas of the goose
# for contour in contours:
#     area = cv2.contourArea(contour)
#     if min_contour_area < area < max_contour_area:
#         # Draw the contour on the original image
#         cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)

# # Display the original image with detected dark areas outlined
# cv2.imshow('Detected Dark Areas', img)

# # Wait for a key press and close all windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(get_head_locations("geese/goose1.jpg"))