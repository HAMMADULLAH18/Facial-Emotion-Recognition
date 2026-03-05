from fer.fer import FER
import cv2

detector = FER(mtcnn=True)
cap = cv2.VideoCapture(0)

print("✅ FER loaded. Reading one frame...")
ret, frame = cap.read()

if ret:
    result = detector.detect_emotions(frame)
    print("Result:", result)
else:
    print("❌ Could not read frame")

cap.release()
cv2.destroyAllWindows()