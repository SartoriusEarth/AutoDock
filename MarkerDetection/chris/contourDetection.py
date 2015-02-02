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

    # Get contours
    contours, hierarchy = cv2.findContours(frame_bw, mode=cv2.cv.CV_RETR_EXTERNAL, method=cv2.cv.CV_CHAIN_APPROX_SIMPLE)

    # Create resultant contour image
    frame_contours = np.zeros((frame_bw.shape[1], frame_bw.shape[0], 3), np.uint8)
    cv2.drawContours(frame_contours, contours, -1, cv2.cv.RGB(255,0,0), 4)

    # Get the center of each contour
    centers = []
    for contour in contours:
        moments = cv2.moments(contour, True)

        # If any moment is 0, discard the entire contour (prevent division by zero)
        if (len(filter(lambda x: x==0, moments.values())) > 0):
            continue

        center = (moments['m10'] / moments['m00'] , moments['m01'] / moments['m00'])

        # Convert to int so we can display it later
        center = map(lambda x: int(round(x)), center)
        centers.append(center)

    # Draw the centers
    for center in centers:
        cv2.circle(frame_contours, tuple(center), 20, cv2.cv.RGB(0,255,255), 2)

    # Show Image
    cv2.imshow('frame_contours', frame_contours)

    # Quit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


captureObj.release()
cv2.destroyAllWindows()