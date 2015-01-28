import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('starry_night.jpg', cv2.IMREAD_GRAYSCALE)

# Initiate the STAR detector
orb = cv2.ORB()

# Find the keypoints with orb
keypoints = orb.detect(image, None)

# Compute the descriptors
keypoints, descriptors = orb.compute(image, keypoints);

# Draw the keypoints onto the image (only location, not size or orientation)
image2 = cv2.drawKeypoints(image, keypoints, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
plt.imshow(image2),plt.show()