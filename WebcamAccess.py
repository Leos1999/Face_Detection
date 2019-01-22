import cv2
import time
import numpy as np
import pytesseract
from PIL import Image

# Path of working folder on Disk
# src_path = "E:/python/"


def get_string(frame):
    # Read image with opencv
    img = frame
    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(img)
    return result


print('--- Start recognizing text from webcam ---')

video = cv2.VideoCapture(0)

a = 0


while True:
    a = a + 1
    check, frame = video.read()
    res = get_string(frame)
    print(" ")
    print(res)
    cv2.imshow("capturing", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()

cv2. destroyAllWindows()
