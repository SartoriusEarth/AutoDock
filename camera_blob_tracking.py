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

        # create masks
        maskBlue = createMask(hsv, lower_blue, upper_blue)
        maskRed1 = createMask(hsv, lower_red1, upper_red1)
        maskRed2 = createMask(hsv, lower_red2, upper_red2)
        maskRed = cv2.bitwise_or(maskRed1, maskRed2)
        maskGreen = createMask(hsv, lower_green, upper_green)
        
        # Bitwise-AND mask and original image
        resultBlue = cv2.bitwise_and(frame, frame, mask=maskBlue)
        resultRed = cv2.bitwise_and(frame, frame, mask=maskRed)
        resultGreen = cv2.bitwise_and(frame, frame, mask=maskGreen)
        
        # locate marker and draw point on image
        blueMarker = locateMarker(maskBlue, wd, ht, blueMarker)
        redMarker = locateMarker(maskRed, wd, ht, redMarker)
        greenMarker = locateMarker(maskGreen, wd, ht, greenMarker)
        radius = 3
        if blueMarker != (-1,-1):
            cv2.circle(frame, blueMarker, radius, (0,255,255), -1, 1)
        if redMarker != (-1,-1):
            cv2.circle(frame, redMarker, radius, (0,255,255), -1, 1)
        if greenMarker != (-1,-1):
            cv2.circle(frame, greenMarker, radius, (0,255,255), -1, 1)
        
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
    
# Returns camera location in world frame
# accepts marker location in world frame
# def estimateCameraPose():
    
    # # Load previously saved camera data
    # with np.load('B.npz') as X:
        # mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

    # # green, blue, red.  green is at the origin
    # worldCoords = np.matrix('0,0,0;0,4,0;0,4,2')
    
    # # TODO replace with detected coordinates
    # pixelCoords = np.matrix('300,300;300,400;200,400')
    
    # solvePnP(worldCoords, pixelCoords
    
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
    
    print "count: " + str(count)
    
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
    
# Run
main()