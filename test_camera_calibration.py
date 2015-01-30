import numpy as np
import cv2
import camera_calibration

def main():
    undistort()

def undistort():
    # Read calibration results from file
    mtx, dist = camera_calibration.read_from_file()
    
    # Create video capture object
    cameraNum = 0;
    cap = cv2.VideoCapture(cameraNum)
    
    # Get video properties
    w = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    
    # Test calibration
    while cap.isOpened():

        ret, img = cap.read()

        # Undistort video
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        newImg = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # TODO crop undistorted image
        
        cv2.imshow('Original Image', img)
        cv2.imshow('Undistorted Image', newImg)
        
        # Quit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
    
# Run program, but not when called from an import statement 
if __name__ == "__main__": 
    main()