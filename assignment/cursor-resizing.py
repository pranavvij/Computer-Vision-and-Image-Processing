import cv2
PATH_TEMPLATE = "/Users/pranavvij/Desktop/cvip/folder/small.png"
PATH_IMG = "/Users/pranavvij/Desktop/cvip/folder/pos_5.jpg"


def resize(img, scale_percent):
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	return resized


img = cv2.imread()
template = cv2.imread(PATH)
es = cv2.matchTemplate(img_gray,template,cv2.TM_SQDIFF_NORMED) 
 

cv2.imshow('image', img)
cv2.waitKey(0)