import numpy as np
import cv2
import math

white = (255,255,255)
black = (0,0,0)

def main():
        
    # Create video capture object
    cameraNum = 0
    cap = cv2.VideoCapture(cameraNum)

    # SSL search parameters
    p = 0.5
    win = 70.0 # MUST BE FLOAT ie 50.0 window size
    
    # Start with unknown location
    marker1 = (-1,-1)
    marker2 = (-1,-1)
    marker3 = (-1,-1)
    counter = 0
    
    while(cap.isOpened()):
    
        # Capture frame-by-frame
        ret, img = cap.read()
        
        # # Display frame early
        # cv2.imshow("image", img);
        
        # get width and height
        h, w = img.shape[:2]
        
        # convert to gray
        imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        thresh = 120
        ret, imgBinary = cv2.threshold(imgGray, thresh, 255, cv2.THRESH_BINARY)

        # print str(counter)
        
        # marker location not known
        if marker1 == (-1,-1):
            # search for marker pattern
            print "global search"
            searchRange = [150,w-150,150,h-150] # [xmin,xmax,ymin,ymax]
            ith = 7
            matches = markerSearch(imgBinary, img, w, h, win, p, ith, searchRange)        
            
            # average clusters of matches
            markers = averageClusters(matches)
            if len(markers) > 0:
                marker1 = markers[0]  
            if len(markers) > 1:
                marker2 = markers[1] 
            if len(markers) > 2:
                marker3 = markers[2] 
            
        # previous marker location known
        else: 
            markersRefined = []
            # for i in range(len(markers)):
                # # refine search
            print "local search"
            r = 10
            searchRange = [marker1[0]-r, marker1[0]+r, marker1[1]-r, marker1[1]+r]
            ith = 5
            matchesRefined = markerSearch(imgBinary, img, w, h, win, p, ith, searchRange)
            markersRefined = markersRefined + averageClusters(matchesRefined)
            
            if marker2 != (-1,-1):
                searchRange = [marker2[0]-r, marker2[0]+r, marker2[1]-r, marker2[1]+r]
                ith = 5
                matchesRefined = markerSearch(imgBinary, img, w, h, win, p, ith, searchRange)
                markersRefined = markersRefined + averageClusters(matchesRefined)
                
            if marker3 != (-1,-1):
                searchRange = [marker3[0]-r, marker3[0]+r, marker3[1]-r, marker3[1]+r]
                ith = 5
                matchesRefined = markerSearch(imgBinary, img, w, h, win, p, ith, searchRange)
                markersRefined = markersRefined + averageClusters(matchesRefined)

            if len(markersRefined) > 0:
                marker1 = markersRefined[0] 
            if len(markersRefined) > 1:
                marker2 = markersRefined[1] 
            if len(markersRefined) > 2:
                marker3 = markersRefined[2] 
            
            print str(markersRefined)
            
            if counter > 10:
                marker1 = (-1,-1)
                counter = 0
                
            if len(markersRefined) == 0:
                counter = counter + 1
            else:
                counter = 0
        
            # Draw cross at averaged match
            for i in range(0,len(markersRefined)):       
                l = 8 # radius of cross hares
                # TODO edge cases
                cv2.line(img, (markersRefined[i][0] - l, markersRefined[i][1]), (markersRefined[i][0] + l, markersRefined[i][1]), (0,0,255), 2)
                cv2.line(img, (markersRefined[i][0], markersRefined[i][1] - l), (markersRefined[i][0], markersRefined[i][1] + l), (0,0,255), 2)
                    
        # while (True):
        # Display frame
        cv2.imshow("binary image", imgBinary);
        cv2.imshow("SSL marker detection", img);
        
        # Quit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break
    
# **** img parameter only needed for drawing****
# skips every ith row and column when searching.  
# min value of ith is 1 max is 5 for reasonable results
# ith = 1 gives highest precision
#
# searchRange = [xmin,xmax,ymin,ymax]
def markerSearch(imgBinary, img, w, h, win, p, ith, searchRange):
    matches = []
    
    # for each row find the best match
    for y in range(searchRange[2],searchRange[3],ith):
        
        # for every pixel in the row except edges
        for x in range(searchRange[0],searchRange[1],ith):
            hpMatch = 0 # horizontal positive direction
            hpMismatch = 0
            
            hnMatch = 0 # horizontal negative direction
            hnMismatch = 0
            
            vpMatch = 0 # vertical positive direction
            vpMismatch = 0
            
            vnMatch = 0 # vertical negative direction
            vnMismatch = 0
            
            # loop through window 
            for i in range(0,int(win)):
                if x+i < w:
                    hpMatch = hpMatch + abs(imgBinary[y,x+i] - imgBinary[y,np.floor(x+i*p)]) / 255
                    hpMismatch = hpMismatch + abs(imgBinary[y,x+i] - imgBinary[y,np.floor(x+i*math.pow(0.5,0.5))]) / 255
                # if x-i > 0:   
                    # hnMatch = hnMatch + abs(imgBinary[y,x-i] - imgBinary[y,np.floor(x-i*p)]) / 255
                    # hnMismatch = hnMismatch + abs(imgBinary[y,x-i] - imgBinary[y,np.floor(x-i*math.pow(0.5,0.5))]) / 255
                if y+i < h:    
                    vpMatch = vpMatch + abs(imgBinary[y+i,x] - imgBinary[np.floor(y+i*p),x]) / 255
                    vpMismatch = vpMismatch + abs(imgBinary[y+i,x] - imgBinary[np.floor(y+i*math.pow(0.5,0.5)),x]) / 255
                # if y-i > 0:    
                    # vnMatch = vnMatch + abs(imgBinary[y-i,x] - imgBinary[np.floor(y-i*p),x]) / 255
                    # vnMismatch = vnMismatch + abs(imgBinary[y-i,x] - imgBinary[np.floor(y-i*math.pow(0.5,0.5)),x]) / 255

            hpM = (hpMismatch - hpMatch) / win # matching function value
            if hpM > 0.3: 
                matches.append((x,y))
                cv2.circle(img, (x,y), 2, (0,255,0), -1)
                print str(hpM)
                
            # hnM = (hnMismatch - hnMatch) / win # matching function value
            # if hnM > 0.3: 
                # matches.append((x,y))
                # cv2.circle(img, (x,y), 2, (0,255,255), -1)
                # print str(hnM)
                
            vpM = (vpMismatch - vpMatch) / win # matching function value
            if vpM > 0.3: 
                matches.append((x,y))
                cv2.circle(img, (x,y), 2, (255,255,0), -1)
                print str(vpM)
                
            # vnM = (vnMismatch - vnMatch) / win # matching function value
            # if vnM > 0.3: 
                # matches.append((x,y))
                # cv2.circle(img, (x,y), 2, (255,0,0), -1)
                # print str(vnM)
    
    return matches
        
def averageClusters(matches):
    result = []
    # average matches taking clusters into account
    # loops through this outer while loop for each cluster
    while len(matches) != 0:
        xsum = 0
        ysum = 0
        # define the current cluster near xgroup and ygroup
        xgroup = matches[0][0]
        ygroup = matches[0][1]
        i = 0
        count = 0
        while i < len(matches):
            # only include matches in the current cluster
            if abs(matches[i][0] - xgroup) < 30 and abs(matches[i][1] - ygroup) < 30:
                xsum = xsum + matches[i][0]
                ysum = ysum + matches[i][1]
                # Delete used matches, when they are all used exit the outer while loop
                del matches[i]
                i = i - 1
                count = count + 1
            i = i + 1
         
        # only use matches if there were several matches in the cluster    
        if count > 0:
            xcen = xsum / count
            ycen = ysum / count
            result.append((xcen,ycen))
    return result
    
main()