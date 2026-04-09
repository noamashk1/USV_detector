import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)

# Set resolution for IMX291
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Read first frame to select ROI
ret, first_frame = cap.read()
if not ret:
    print("Camera error")
    exit()

# Select ROI with mouse - Press ENTER when done
print("Select ROI with mouse and press ENTER")
r = cv2.selectROI("Select ROI", first_frame)
x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])
cv2.destroyWindow("Select ROI")

# Process first frame for comparison
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_roi = first_gray[y:y+h, x:x+w]
first_roi = cv2.GaussianBlur(first_roi, (21, 21), 0)

print("Monitoring... Press 'q' to stop")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process current frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[y:y+h, x:x+w]
    roi_blurred = cv2.GaussianBlur(roi, (21, 21), 0)

    # Calculate difference
    frame_delta = cv2.absdiff(first_roi, roi_blurred)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Count motion pixels
    motion_amount = np.sum(thresh) / 255

    # Trigger action
    if motion_amount > 500: 
        cv2.putText(frame, "MOTION DETECTED!", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Visual feedback
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Lab Feed", frame)
    cv2.imshow("Motion Mask", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()