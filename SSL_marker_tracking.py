import numpy as np
import cv2

# Create concentric SSL image
w = 200;
h = 200;
marker = np.zeros((h,w,3), np.uint8 );
center = (100,100)
white = (255,255,255)
black = (0,0,0)

cv2.circle(marker, center, 71, white, -1)
cv2.circle(marker, center, 50, black, -1)
cv2.circle(marker, center, 35, white, -1)
cv2.circle(marker, center, 25, black, -1)
cv2.circle(marker, center, 17, white, -1)
cv2.circle(marker, center, 12, black, -1)
cv2.circle(marker, center, 9, white, -1)
cv2.circle(marker, center, 6, black, -1)
cv2.circle(marker, center, 4, white, -1)
cv2.circle(marker, center, 3, black, -1)
cv2.circle(marker, center, 2, white, -1)

# convert to gray
markerGray = cv2.cvtColor(marker,cv2.COLOR_BGR2GRAY)

p = 0.5
 
while (True):

    # for each row find the best match
    for j in range(0,h,5):
        result = np.zeros(w-1,np.int64)
        ri = 0; # result index
        
        # for every pixel in the row except edges
        for x in range(0,w-100):
            sumMatch = 0
            for i in range(0,50):
                 sumMatch = sumMatch + abs(markerGray[x+i,j] - markerGray[np.floor(x+i*p),j]) 
            
            sumMismatch = 0
            for i in range(0,50):
                sumMismatch = sumMismatch + abs(markerGray[x+i,j] - markerGray[np.floor(x+i*0.7071),j]) 
            
            result[ri] = (sumMismatch - sumMatch) / 50;
            ri = ri + 1
        
        # find index of best result
        max = 0
        best = 0
        for i in range(0,w-1):
            if result[i] > 1:
                max = result[i]
                best = i
                
        print str(result[100])
        
        # draw best match for each for
        cv2.circle(marker, (best,j), 2, (0,255,0), -1)

    # Display frame
    cv2.imshow("SSL marker detection", marker);
    
    # Quit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break