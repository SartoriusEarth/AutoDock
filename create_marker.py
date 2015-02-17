import numpy as np
import cv2
import math

white = (255,255,255)
black = (0,0,0)

def main():
    # img attributes
    w = 800
    h = 800
    center = (w/2,h/2)
    scale = 500
    
    # Create blank white image
    img = np.ones((h,w,3), np.uint8)
    for j in range(0,h-1):
        for i in range(0,w-1):
            img[j,i] = white

    # Draw SSL on image
    createSSL(img, center, scale)
    # createSSL(img, (200,600), 200)
    # createSSL(img, (600,600), 200)

    # write image to file
    cv2.imwrite("single_marker.jpeg",img);
        
    # Dispaly image
    while (1):
        # Display frame
        cv2.imshow("SSL marker", img)
        
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