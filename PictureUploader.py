
import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
cap = cv2.VideoCapture(0)
i = 0
while True:
    # Capture frame by frame
    ret, frame = cap.read()
    # img_item = "l.png"
    image_item = os.path.join(image_dir, str(i) + ".png")
    cv2.imwrite(image_item, frame)
    cv2.imshow('frame', frame)
    i = i + 1
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Releasing the captured
cap.release()
cv2.destroyAllWindows()
