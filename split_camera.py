import cv2

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Set the resolution of the captured video (e.g., 1920x1080)
desired_width = 1920*2
desired_height = 1080

cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Get the actual resolution of the camera after setting
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Resolution set to: {actual_width}x{actual_height}")

# Define width for the split screen
half_width = actual_width // 2

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Split the frame into left and right halves
    left_half = frame[:, :half_width]
    right_half = frame[:, half_width:]

    # Display the two halves in separate windows
    cv2.imshow("Left Half", left_half)
    cv2.imshow("Right Half", right_half)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
