import numpy as np
import cv2

if __name__ == '__main__':
    # Choose primary camera and create video capture object
    cameraIndex = 0
    captureObj = cv2.VideoCapture(cameraIndex)

    while True:
        # Get current frame
        # ret - true if frame read correctly, false otherwise
        # frame - current frame (is an image matrix)
        ret, frame = captureObj.read()

        cv2.imshow('frame', frame)

        # Quit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    captureObj.release()
    cv2.destroyAllWindows()