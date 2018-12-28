import cv2
import numpy as np

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged


img_rgb = cv2.imread('/Users/pranavvij/Desktop/task3/pos_15.jpg', cv2.IMREAD_GRAYSCALE);
template = cv2.imread('/Users/pranavvij/Desktop/task3/small.jpg', cv2.IMREAD_GRAYSCALE) 

canny_img = auto_canny(img_rgb)
canny_template_img = auto_canny(template)

w, h = canny_template_img.shape[::-1]

res = cv2.matchTemplate(canny_img, canny_template_img, cv2.TM_SQDIFF_NORMED) 

threshold = 0.8
loc = np.where( res >= threshold) 

print(loc)

for pt in zip(*loc[::-1]): 
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2) 

cv2.imshow('Detected',img_rgb) 
cv2.imshow('laplacian', canny_template_img)
cv2.imshow('canny_img', canny_img)
cv2.waitKey(0)

