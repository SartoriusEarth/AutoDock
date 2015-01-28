import numpy as np
import cv2
import glob

CHESSBOARD_WIDTH = 9
CHESSBAORD_HEIGHT = 7

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHESSBOARD_WIDTH*CHESSBAORD_HEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD_WIDTH,0:CHESSBAORD_HEIGHT].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Create video capture object
cameraNum = 0;
cap = cv2.VideoCapture(cameraNum)

# Get video properties
w = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

chessSamples = 0

#Collect imgpoints
while cap.isOpened():
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_WIDTH,CHESSBAORD_HEIGHT),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        chessSamples = chessSamples + 1
        print str(chessSamples)
    
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (CHESSBOARD_WIDTH,CHESSBAORD_HEIGHT), corners, ret)
    
    cv2.imshow('img',img)

    # Quit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

# calibrate camera
print "calibrating"
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print "finished"

# mtx = np.matrix('927.85,0,385.35;0,813.87,4.322;0,0,1')
# dist = np.matrix('0.203, -0.0713, -0.054324, 0.013143, 0.007245')

print "mtx" + str(mtx)
print "dist" + str(dist)
# # print "rvecs" + str(rvecs)
# # print "tvecs" + str(tvecs)



# # Test calibration
# while cap.isOpened():

    # ret, img = cap.read()

    # # Undistort video
    # newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # # w,h,d = dst.shape
    # # print str(w) + " " + str(h)
    
    # # crop the image
    # x,y,w,h = roi
    # newImg = dst[y:y+h, x:x+w]

    # cv2.imshow('img',dst)
    
    # # Quit on 'q' press
    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break
        
# cv2.destroyAllWindows()