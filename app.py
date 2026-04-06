import cv2
from ultralytics import YOLO

# Model load karein (Ensure best.tflite is in the same folder)
model = YOLO('best.tflite', task='detect')

# Camera ya Video input (0 for webcam)
cap = cv2.VideoCapture(0)

print("Police AI Model Started... Press 'q' to stop.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Detection shuru
    results = model(frame, imgsz=320, conf=0.25)

    # Results ko screen par dikhana
    annotated_frame = results[0].plot()
    cv2.imshow("DigitalMadhu Police AI", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
