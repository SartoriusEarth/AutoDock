import numpy as np
import cv2

## Needs some work, a little slow.

# Returns cosine of a triangle defined by the three points.
#      2
#      *
#     /|
#    / |
#   /  |
#  /   |
# *----*
# 1    0
def cos(point0, point1, point2):
    d1, d2 = (point0 - point1).astype('float'), (point2 - point1).astype('float')
    return abs( np.dot(d1,d2) / np.sqrt( np.dot(d1,d1) * np.dot(d2,d2) ) )

# Find the squares in an image.
def find_squares(img):
    # Remove noise via averaging
    img = cv2.GaussianBlur(img, (5,5), 0)

    squares = []
    # split is slow; if necessary change to use indexing
    for gray in cv2.split(img):
        for threshold in xrange(0, 255, 26):
            if threshold == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                ret, bin = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

            contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour_length = cv2.arcLength(contour, True)
                contour = cv2.approxPolyDP(contour, 0.02*contour_length, True)
                if len(contour) == 4 and cv2.contourArea(contour) > 1000 and cv2.isContourConvex(contour):
                    contour = contour.reshape(-1, 2)
                    max_cos = np.max([cos( contour[i], contour[(i+1) % 4], contour[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(contour)

    return squares



if __name__ == '__main__':
    # Choose primary camera and create video capture object
    cameraIndex = 0
    captureObj = cv2.VideoCapture(cameraIndex)

    while True:
        # Get current frame
        # ret - true if frame read correctly, false otherwise
        # frame - current frame (is an image matrix)
        ret, frame = captureObj.read()

        squares = find_squares(frame)
        cv2.drawContours( frame, squares, -1, (0, 255, 0), 3 )
        
        cv2.imshow('frame', frame)

        # Quit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    captureObj.release()
    cv2.destroyAllWindows()