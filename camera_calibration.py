USAGE = """

This program calibrates a camera

Requires extra parameter '1' to override existing camera calibration

Usage:
python camera_calibration.py
python camera_calibration.py 1 -to write to file

Example:
python camera_calibration.py 1

"""

import sys
import numpy as np
import cv2
import test_camera_calibration 

CHESSBOARD_WIDTH = 9
CHESSBAORD_HEIGHT = 7

def main():
    # Get command line arguments
    print USAGE
    write = False
    if len(sys.argv) == 2:
        if int(sys.argv[1]) == 1:
            write = True

    # termination criteria for cornerSubPix function
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

    # Initialize counter for successful samples
    chessSamples = 0

    #Collect imgpoints
    while cap.isOpened():
        # Read image and convert to grayscale for processing
        ret, img = cap.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_WIDTH,CHESSBAORD_HEIGHT),None)

        # If chessboard corners found, add object points, image points (after refining them)
        if ret == True:
            # Increment counter for successful samples
            chessSamples = chessSamples + 1
        
            # Include fewer samples in calibration calculation to avoid excessive computation time
            if chessSamples % 5 == 0:
                print str(chessSamples/5)
                
                # Add chessboard points measured in the world
                objpoints.append(objp)

                # Refines corner detection
                cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                
                # Add detected corners pixel location
                imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (CHESSBOARD_WIDTH,CHESSBAORD_HEIGHT), corners, ret)

        # Display the video feed
        cv2.imshow('img',img)

        # Limit number of samples accumulated to avoid excessive computation times
        if chessSamples/5 > 50:
            break
        
        # Quit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    # calibrate camera
    print "calibrating"
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    print "finished\n"
    
    # Write calibration results to file so they can be retrieved easy later
    if write:
        write_to_file(mtx, dist)

    # Test calibration
    test_camera_calibration.undistort()

# Write camera calibration data to a file
def write_to_file (mtx, dist):
    # Open a file is write mode
    f = open('cam_cal.txt', 'w')
    
    # Write mtx and dist matrix
    np.savez(f, mtx, dist)
    
    # Close file so it can be read from later
    f.close()
    
    print "calibration results written to file cam_cal.txt\n"
    
# Read camera calibration read from file
def read_from_file ():
    # Open file to read from
    f = open('cam_cal.txt', 'r')
    
    # Read file
    npzfile = np.load(f)
    mtx = npzfile['arr_0']
    dist = npzfile['arr_1']
    
    # Close file
    f.close()
    
    print "Read camera calibration data file"
    
    return mtx, dist
    
# Run program, but not when called from an import statement 
if __name__ == "__main__": 
    main()