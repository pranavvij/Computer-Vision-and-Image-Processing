import cv2
import numpy as np

PATH_TEMPLATE = "/Users/pranavvij/Desktop/cvip/folder/small.png"
PATH_IMG = "/Users/pranavvij/Desktop/cvip/folder/pos_5.jpg"


def resize(img, scale_percent):
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	return resized

def get_laplacian(img):
	return cv2.Laplacian(img, cv2.CV_8U)

def get_gausian(img):
	return cv2.GaussianBlur(img,(5,5),0)

def get_canny(img,start,end):
	return cv2.Canny(img, start, end)

def show(img):
	cv2.imshow('image', img)
	cv2.waitKey(0)

for i in range(0, 1000):
	for j in range(i, 1000):
		for k in range(0, 1000):
			for l in range(k, 1000):
				print(i,j,k,l)
				img = cv2.imread(PATH_IMG)
				img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
				img_gray = get_gausian(img_gray)
				img_gray = get_canny(img_gray,i, j)

				template = cv2.imread(PATH_TEMPLATE)
				template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY);
				template = get_gausian(template)
				template = get_canny(template,k, l)

				#show(img_gray)
				#show(template)

				scale_percent = 40

				for scale in range(scale_percent, scale_percent + 21):
					img_copy =  img_gray.copy()
					template_new = resize(template, scale)
					res = cv2.matchTemplate(img_copy, template_new, cv2.TM_CCORR_NORMED) 
					# print("=========================")
					# print(scale)
					# print(template_new.shape)
					# print("=========================")
					w, h = template_new.shape[::-1] 
					threshold = 0.8
					loc = np.where( res >= threshold)  
					l_len = len(loc[0])
					if l_len > 0:
						print(i,j,k,l)
						print(scale)
					# for pt in zip(*loc[::-1]): 
					# 	cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2) 

					#cv2.imshow('image', img)
					#cv2.waitKey(0)


