import cv2
import numpy as np

# Load the image
image = cv2.imread('goose_image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny
edges = cv2.Canny(blurred, 50, 150)

# Detect circles using HoughCircles
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                           param1=100, param2=30, minRadius=5, maxRadius=30)

# Ensure some circles were found
if circles is not None:
  circles = np.round(circles[0, :]).astype("int")
  
  for (x, y, r) in circles:
    # Draw the circle in the output image
    cv2.circle(image, (x, y), r, (0, 255, 0), 4)
    # Draw a rectangle at the center of the circle
    cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

# Show the output image
cv2.imshow("Detected Eyes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()