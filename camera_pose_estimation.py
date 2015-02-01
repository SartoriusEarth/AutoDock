USAGE = """

This program uses color object segmentation to track a marker

Usage:
python camera_pose_estimation.py
python camera_pose_estimation.py "camera number"
python camera_pose_estimation.py "camera number" "output file basename"

Example:
python camera_pose_estimation.py 1 underwaterCam

"""

from time import localtime, strftime
import numpy as np
import cv2
import sys

RECORD = False;

def main():
    # Get command line arguments
    print USAGE
    if len(sys.argv) <= 1:
        cameraNum = 0
        basename = "camera"
    elif len(sys.argv) == 2:
        cameraNum = int(sys.argv[1])
        basename = "camera"
    elif len(sys.argv) == 3:
        cameraNum = int(sys.argv[1])
        basename = str(sys.argv[2])
        
    # Create video capture object
    cap = cv2.VideoCapture(cameraNum)

    # Get video properties
    wd = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    ht = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps = int(10)
    #print "FPS: " + str(cap.get(cv2.cv.CV_CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
    outputFileName = basename + str(cameraNum) + '_' + strftime('%Y%m%d_%H %M %S', localtime())
    
    if RECORD:
        out = cv2.VideoWriter(outputFileName + '.avi',fourcc, fps, (wd,ht), True)
    
    # start with unknown position
    blueMarker = (-1,-1)
    redMarker = (-1,-1)
    greenMarker = (-1,-1)
    blackMarker = (-1,-1)

    while(cap.isOpened()):
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
        
        # define range of red color in HSV
        lower_red1 = np.array([140,100,100])
        upper_red1 = np.array([255,255,255])
        lower_red2 = np.array([0,100,100])
        upper_red2 = np.array([30,255,255])
        
        # define range of red color in HSV
        lower_green = np.array([60,20,20])
        upper_green = np.array([100,255,255])
        
        # define range of black color in HSV
        lower_black = np.array([0,0,0])
        upper_black = np.array([255,255,40])

        # create masks
        maskBlue = createMask(hsv, lower_blue, upper_blue)
        maskRed1 = createMask(hsv, lower_red1, upper_red1)
        maskRed2 = createMask(hsv, lower_red2, upper_red2)
        maskRed = cv2.bitwise_or(maskRed1, maskRed2)
        maskGreen = createMask(hsv, lower_green, upper_green)
        maskBlack = createMask(hsv, lower_black, upper_black)
        
        # Bitwise-AND mask and original image
        resultBlue = cv2.bitwise_and(frame, frame, mask=maskBlue)
        resultRed = cv2.bitwise_and(frame, frame, mask=maskRed)
        resultGreen = cv2.bitwise_and(frame, frame, mask=maskGreen)
        resultBlack = cv2.bitwise_and(frame, frame, mask=maskBlack)
        
        # locate marker and draw point on image
        blueMarker = locateMarker(maskBlue, wd, ht, blueMarker)
        redMarker = locateMarker(maskRed, wd, ht, redMarker)
        greenMarker = locateMarker(maskGreen, wd, ht, greenMarker)
        blackMarker = locateMarker(maskBlack, wd, ht, blackMarker)
        radius = 6
        if blueMarker != (-1,-1):
            cv2.circle(frame, blueMarker, radius, (0,255,255), -1, 1)
        if redMarker != (-1,-1):
            cv2.circle(frame, redMarker, radius, (0,255,255), -1, 1)
        if greenMarker != (-1,-1):
            cv2.circle(frame, greenMarker, radius, (0,255,255), -1, 1)
        if blackMarker != (-1,-1):
            cv2.circle(frame, blackMarker, radius, (0,255,255), -1, 1)
            
        # Camera matrices from camera calibration
        mtx = np.matrix('927.85,0,385.35;0,813.87,4.322;0,0,1')
        dist = np.matrix('0.203, -0.0713, -0.054324, 0.013143, 0.007245')
        
        # Define marker points (3D)
        worldCoord = np.matrix('0,0,0;5,0,0;5,3,0;0,3,0', np.float32)
        
        # Markers found by camera (2D)
        markers = [greenMarker, blueMarker, redMarker, blackMarker]
        imageCoord = np.asmatrix(markers, np.float32)
    
        axis = np.float32([[1,0,0], [0,1,0], [0,0,1]]).reshape(-1,3)

        if blueMarker != (-1,-1) and redMarker != (-1,-1) and greenMarker != (-1,-1) and blackMarker != (-1,-1):
            # Find the rotation and translation vectors.
            rvecs, tvecs, inliers = cv2.solvePnPRansac(worldCoord, imageCoord, mtx, dist)
            
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            
            # print "rvecs: " + str(rvecs)
            # print "tvecs: " + str(tvecs)
            
            computeCameraCoord(rvecs, tvecs)
            
            # Draw axis
            markerOrigin = tuple(np.squeeze(np.asarray(imageCoord[0])))
            cv2.line(frame, markerOrigin, tuple(np.squeeze(np.asarray(imgpts[0]))), (255,0,0), 5)  # blue
            cv2.line(frame, markerOrigin, tuple(np.squeeze(np.asarray(imgpts[1]))), (0,255,0), 5)  # red
            cv2.line(frame, markerOrigin, tuple(np.squeeze(np.asarray(imgpts[2]))), (0,0,255), 5)  # green
        
        if RECORD:
            # Write frame to file
            out.write(frame)
            
        # Display  frame
        cv2.imshow(outputFileName, frame)
        
        # Quit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything
    cap.release()
    if RECORD:
        out.release()
    cv2.destroyAllWindows()
    
    
# this function calculates average position of the color marker 
# prevMark value of (-1,-1) means marker has not been found 
def locateMarker(mask, wd, ht, prevMark): 
    searchSize = 50
    count = 1 # can't divide by zero so must start at one
    xSum = 0
    ySum = 0
    
    # If previous marker location is not known search whole image
    if prevMark[0] == -1 or prevMark[1] == -1:
        xRng = (0,wd)
        yRng = (0,ht)
    # When previous marker location is know search a smaller area specified by searchSize
    else:
        # take previous marker location and search nearby
        xRng = (prevMark[0]-searchSize,prevMark[0]+searchSize)
        yRng = (prevMark[1]-searchSize,prevMark[1]+searchSize)
        # limit search area to valid frame
        if xRng[0] < 0:
            xRng = (0,xRng[1])
        if xRng[1] > wd:
            xRng = (xRng[0],wd-1)
        if yRng[0] < 0:
            yRng = (0,yRng[1])
        if yRng[1] > ht:
            yRng = (yRng[0],ht-1)

    # Search x and y range for object 
    # only look at every 'step' pixel because mask is coarse
    step = 3
    for i in range(yRng[0],yRng[1],step):
        for j in range(xRng[0],xRng[1],step):
            if mask[i,j] > 0:
                # # if marker position known
                # if prevMark[0] != -1 and preveMark[1] != -1:
                count = count + 1
                xSum = xSum + j
                ySum = ySum + i
                
    # Find average of pixels corresponding to blueMarker of object
    xAvg = xSum / count
    yAvg = ySum / count
    
    #print "count: " + str(count)
    
    # Object not found unless a large chunk is seen
    if count < 50:
        xAvg = -1
        yAvg = -1
    return (xAvg,yAvg)  

# creates a color mask based on HSV upper and lower limits
# Uses opening to remove noise in the image
def createMask(hsv, lower, upper):
    # Threshold the HSV image to get only blue colors
    maskBlue = cv2.inRange(hsv, lower, upper)
    
    # Dilate then erode mask to remove noise 
    kernel = np.ones((7,7),np.uint8)
    maskBlue = cv2.morphologyEx(maskBlue, cv2.MORPH_OPEN, kernel) 
    
    return maskBlue
   
# Compute the cameras coordinates relative to the marker origin
# takes a rotation vector and translation vector
# returns cameras coordinates relative 
def computeCameraCoord(rvec, tvec):
    # compute rotation matrix from rotation vector
    rmat, jacobian = cv2.Rodrigues(rvec)
    print "Camera rotation matrix:\n" + str(rmat)
    
    # Transpose rotation matrix and multiply its negative by the translation vector
    rmat_trans = rmat.transpose()
    t = -rmat_trans*np.asmatrix(tvec)
    print "camera position:\n" + str(t)
    
# Run
main()