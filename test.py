
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

img = cv2.imread(glob.glob("imgs/*")[0])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Using the Canny filter to get contours
edges = cv2.Canny(gray, 20, 30)
# Using the Canny filter with different parameters
edges_high_thresh = cv2.Canny(gray, 60, 90)
# Stacking the images to print them together
# For comparison
images = np.hstack((gray, edges, edges_high_thresh))

# Display the resulting frame
cv2.imshow('Frame', images)
cv2.waitKey()
