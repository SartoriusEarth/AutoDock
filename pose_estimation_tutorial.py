USAGE = """

This program records and displays video feed from a USB camera

Usage:
python pose_estimation_tutorial.py

Example:
python pose_estimation_tutorial.py 

"""

from time import localtime, strftime
import numpy as np
import cv2
import sys

CHESSBOARD_WIDTH = 3
CHESSBAORD_HEIGHT = 3
    
def main():
    # Create video capture object
    cameraNum = 0
    cap = cv2.VideoCapture(cameraNum)
    
    # Camera matrices from camera calibration
    mtx = np.matrix('927.85,0,385.35;0,813.87,4.322;0,0,1')
    dist = np.matrix('0.203, -0.0713, -0.054324, 0.013143, 0.007245')

    while(cap.isOpened()):
        
        # Capture frame-by-frame
        ret, img = cap.read()
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Define chessboard plane
        objp = np.zeros((CHESSBOARD_WIDTH*CHESSBAORD_HEIGHT,3), np.float32)
        objp[:,:2] = np.mgrid[0:CHESSBOARD_WIDTH,0:CHESSBAORD_HEIGHT].T.reshape(-1,2)
        
        # define 2x2 checkerboard
        smallObj = np.zeros((4,3), np.float32)
        smallObj[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)
        # smallObj = smallObj[0:3]

        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_WIDTH,CHESSBAORD_HEIGHT),None)
        
        cv2.drawChessboardCorners(img, (CHESSBOARD_WIDTH,CHESSBAORD_HEIGHT), corners, ret)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            
            #smallObjCorners = np.zeros((1,2), np.float32)
            smallObjCorners = corners[0:4]
            smallObjCorners[2:4] = corners[CHESSBOARD_WIDTH:CHESSBOARD_WIDTH+2]
            
            print str(smallObj)
            print "corners: \n" + str(smallObjCorners[0])
            
            # draw corners used for pose estimation
            radius = 7
            cv2.circle(img, tuple(np.squeeze(np.asarray(smallObjCorners[0]))), radius, (0,255,255), -1, 1)
            cv2.circle(img, tuple(np.squeeze(np.asarray(smallObjCorners[1]))), radius, (0,255,255), -1, 1)
            cv2.circle(img, tuple(np.squeeze(np.asarray(smallObjCorners[2]))), radius, (0,255,255), -1, 1)
            cv2.circle(img, tuple(np.squeeze(np.asarray(smallObjCorners[3]))), radius, (0,255,255), -1, 1)


            # Find the rotation and translation vectors.
            rvecs, tvecs, inliers = cv2.solvePnPRansac(smallObj, smallObjCorners, mtx, dist)

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

            # Draw axis
            corner = tuple(corners[0].ravel())
            # print str(corner)
            cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
            cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
            cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        
        # Display  frame
        cv2.imshow('img',img)
        
        # Quit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything
    cap.release()
    cv2.destroyAllWindows()

main()