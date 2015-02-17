import numpy as np
import cv2
import math

white = (255,255,255)
black = (0,0,0)

def main():
    # Create concentric SSL image
    w = 200;
    h = 200;
    marker = np.zeros((h,w,3), np.uint8 );
    center = (100,100)
    
    createSSL(marker, center, 200)

    # convert to gray
    markerGray = cv2.cvtColor(marker,cv2.COLOR_BGR2GRAY)
    ret,markerBinary = cv2.threshold(markerGray, 127, 255, cv2.THRESH_BINARY)

    p = 0.5
    win = 50.0 # MUST BE FLOAT ie 50.0 window size

    # for each row find the best match
    for j in range(0,h,5):
        result = np.zeros(w-1,np.float)
        ri = 0; # result index
        
        # for every pixel in the row except edges
        for x in range(0,w-int(win)):
            sumMatch = 0
            for i in range(0,int(win)):
                 sumMatch = sumMatch + abs(markerBinary[x+i,j] - markerBinary[np.floor(x+i*p),j]) / 255
            
            sumMismatch = 0
            for i in range(0,int(win)):
                sumMismatch = sumMismatch + abs(markerBinary[x+i,j] - markerBinary[np.floor(x+i*0.7071),j]) / 255

            result[ri] = (sumMismatch - sumMatch) / win # not sure what the divisor value should be maybe 150
            ri = ri + 1
        
        # find index of best result
        max = 0.0
        best = 0
        for i in range(0,w-1):
            if result[i] > max and result[i] > 0.3:
                max = result[i]
                best = i
        
        # draw best match for each for
        cv2.circle(marker, (best,j), 2, (0,255,0), -1)
        
        print str(result[best])

      
    while (True):
        # Display frame
        cv2.imshow("SSL marker detection", marker);
        
        # Quit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
# Create Concentric SSL pattern            
def createSSL(img, center, scale):
    for i in range(0,10):
        cv2.circle(img, center, int(math.pow(0.5,i+0.5)*scale), black, -1)
        cv2.circle(img, center, int(math.pow(0.5,i+1)*scale), white, -1)
    
main()