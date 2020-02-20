import cv2
import numpy as np
from skimage import io
from skimage.util import img_as_ubyte
from skimage.exposure import histogram
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.viewer import ImageViewer

showImage = lambda image: ImageViewer(image).show()

im = io.imread('data/Con1.tif')
im = im[10]
hist, hist_centers = histogram(im)

showImage(im)

# # canny edge detection
# edges = canny(im/255.)
#
# # [if] background smooth, almost all edges found at boundary of object or inside
# fill_im = ndi.binary_fill_holes(edges)
#
# # remove objects smaller than threshold
# label_objects, nb_labels = ndi.label(fill_im)
# sizes = np.bincount(label_objects.ravel())
# mask_sizes = sizes > 20
# mask_sizes[0] = 0
# im_cleaned = mask_sizes[label_objects]

markers = np.zeros_like(im)
markers[im < 30] = 1
markers[im > 150] = 2
elevation_map = sobel(im)

# showImage(elevation_map)


image_input = img_as_ubyte(im)

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


