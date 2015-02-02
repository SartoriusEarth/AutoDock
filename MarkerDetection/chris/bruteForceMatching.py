import numpy as np
import cv2
from matplotlib import pyplot as plot
from drawMatches import drawMatches

queryImg = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
trainImg = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

# Initiate SIFT detector
orb = cv2.ORB()

# Find the keypoints and descriptors with SIFT
keypointsQuery, descriptorsQuery = orb.detectAndCompute(queryImg, None)
keypointsTrain, descriptorsTrain = orb.detectAndCompute(trainImg, None)

# Create a BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(descriptorsQuery, descriptorsTrain)

# Sort based on distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches
matchedImg = drawMatches(queryImg, keypointsQuery, trainImg, keypointsTrain, matches[:15])
plot.imshow(matchedImg),plot.show()