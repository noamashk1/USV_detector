import cv2
import numpy as np
import time

GPIO_TTL = 17  # BCM numbering
PULSE_WIDTH_S = 0.05  # short TTL pulse (10 ms)
MOTION_THRESHOLD = 500

gpio_handle = None
try:
    import lgpio

    gpio_handle = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_output(gpio_handle, GPIO_TTL)
    lgpio.gpio_write(gpio_handle, GPIO_TTL, 0)
    print(f"TTL output enabled on GPIO{GPIO_TTL}")
except ImportError:
    print("lgpio not available — TTL output disabled")


def ttl_pulse():
    if gpio_handle is None:
        return
    lgpio.gpio_write(gpio_handle, GPIO_TTL, 1)
    time.sleep(PULSE_WIDTH_S)
    lgpio.gpio_write(gpio_handle, GPIO_TTL, 0)


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

prev_motion = False
try:
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
        motion_detected = motion_amount > MOTION_THRESHOLD

        # Trigger action + TTL pulse on rising edge (once per motion event)
        if motion_detected:
            cv2.putText(frame, "MOTION DETECTED!", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            if not prev_motion:
                ttl_pulse()

        prev_motion = motion_detected

        # Visual feedback
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Lab Feed", frame)
        cv2.imshow("Motion Mask", thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    if gpio_handle is not None:
        lgpio.gpio_write(gpio_handle, GPIO_TTL, 0)
        lgpio.gpiochip_close(gpio_handle)
