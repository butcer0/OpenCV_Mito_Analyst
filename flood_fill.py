import cv2
import numpy as np

# Read image
IMAGE_INPUT_NICKLE = cv2.imread('data/nickle.png', cv2.IMREAD_GRAYSCALE)
IMAGE_INPUT_MITO = cv2.imread('data/Con1.tif', cv2.IMREAD_UNCHANGED)
image_input = IMAGE_INPUT_NICKLE

# Threshold.
# Set values equal to or above 220 to 0.
# Set values below 220 to 255

threshold, image_thresholded = cv2.threshold(image_input, 220, 255, cv2.THRESH_BINARY_INV)

# Copy the thresholded image
image_floodfill = image_thresholded.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = image_thresholded.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(image_floodfill, mask, (0, 0), 255);

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(image_floodfill)

# Combine the two images to get the foreground.
im_out = image_thresholded | im_floodfill_inv

# Display images.
cv2.imshow("Input Image", image_input)
cv2.imshow("Thresholded Image", image_thresholded)
cv2.imshow("Floodfilled Image", image_floodfill)
cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
cv2.imshow("Foreground", im_out)
cv2.waitKey(0)
