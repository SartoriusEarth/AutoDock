import numpy as np
import cv2

# Choose primary camera and create video capture object
cameraIndex = 0
captureObj = cv2.VideoCapture(cameraIndex)

while True:
    # Get current frame
    # ret - true if frame read correctly, false otherwise
    # frame - current frame (is an image matrix)
    ret, frame = captureObj.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Remove noise with Gaussian blur
    sigmaX = 5
    sigmaY = 5
    frame = cv2.GaussianBlur(frame, (sigmaX,sigmaY), 0)

    # Adaptive Thresholding
    block_size = 11
    C = 2
    frame_bw = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)

    # Get contours - TODO finish contours and add them to image
    contours, hierarchy = cv2.findContours(frame_bw, mode=cv2.cv.CV_RETR_EXTERNAL, method=cv2.cv.CV_CHAIN_APPROX_SIMPLE)

    # Show Image
    cv2.imshow('frame_bw', frame_bw)

    # Quit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


captureObj.release()
cv2.destroyAllWindows()