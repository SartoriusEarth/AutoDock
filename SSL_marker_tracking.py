import numpy as np
import cv2
import math

white = (255,255,255)
black = (0,0,0)

def main():
    # Create concentric SSL image
    w = 200;
    h = 200;
    img = np.zeros((h,w,3), np.uint8 );
    center = (100,100)
    
    p = 0.5
    win = 50.0 # MUST BE FLOAT ie 50.0 window size
    
    # create image of marker
    createSSL(img, p, center, 200) 
       
    # Load image
    img = cv2.imread("marker_sample_3.jpg")
    h, w = img.shape[:2]  
    img = cv2.resize(img, (w/4,h/4));
    h, w = img.shape[:2]
    # img[y,x]
    
    # convert to gray
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = 130
    ret, imgBinary = cv2.threshold(imgGray, thresh, 255, cv2.THRESH_BINARY)

    matches = []
    
    # for each row find the best match
    for y in range(50,h-50,5):
        
        # for every pixel in the row except edges
        for x in range(0,w-1,5):
            sumMatch = 0
            sumMismatch = 0
            for i in range(0,int(win)):
                if x+i < w:
                    sumMatch = sumMatch + abs(imgBinary[y,x+i] - imgBinary[y,np.floor(x+i*p)]) / 255

                    sumMismatch = sumMismatch + abs(imgBinary[y,x+i] - imgBinary[y,np.floor(x+i*0.7071)]) / 255

            m = (sumMismatch - sumMatch) / win # matching function value
            if m > 0.3: 
                matches.append((x,y))
                cv2.circle(img, (x,y), 2, (0,255,0), -1)
                print str(m)
    
    # average matches taking clusters into account
    while len(matches) != 0:
        print "new group"
        xsum = 0
        ysum = 0
        xgroup = matches[0][0]
        ygroup = matches[0][1]
        i = 0
        count = 0
        while i < len(matches):
            print str(i)
            print str(matches)
            if abs(matches[i][0] - xgroup) < 30 and abs(matches[i][1] - ygroup) < 30:
                print "add " + str(matches[i])
                xsum = xsum + matches[i][0]
                ysum = ysum + matches[i][1]
                del matches[i]
                i = i - 1
                count = count + 1
            i = i + 1
        xcen = xsum / count
        ycen = ysum / count
        
        # Draw cross at averaged match
        l = 4 # radius of cross hares
        # TODO edge cases
        cv2.line(img, (xcen - l, ycen), (xcen + l, ycen), (0,0,255), 1)
        cv2.line(img, (xcen, ycen - l), (xcen, ycen + l), (0,0,255), 1)
                
    while (True):
        # Display frame
        cv2.imshow("binary image", imgBinary);
        cv2.imshow("SSL marker detection", img);
        
        # Quit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
# Create Concentric SSL pattern            
def createSSL(img, p, center, scale):
    for i in range(0,10):
        cv2.circle(img, center, int(math.pow(p,i+0.5)*scale), black, -1)
        cv2.circle(img, center, int(math.pow(p,i+1)*scale), white, -1)
    
main()