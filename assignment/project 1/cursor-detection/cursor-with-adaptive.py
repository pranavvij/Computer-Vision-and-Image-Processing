import cv2
import numpy as np


def resize(img, scale_percent):
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	return resized

def get_laplacian(img):
	return cv2.Laplacian(img, cv2.CV_8U,ksize = 3)

def get_gausian(img):
	return cv2.GaussianBlur(img,(3,3),0)

def get_canny(img,start,end):
	return cv2.Canny(img, start, end)

def show(img):
	cv2.imshow('image', img)
	cv2.waitKey(0)

def image_without_bg(img):
	height, width = img.shape[:2]
	mask = np.zeros(img.shape[:2],np.uint8)

	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)

	#Hard Coding the Rect… The object must lie within this rect.
	rect = (0,2,width,height)
	cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
	mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	img1 = img*mask[:,:,np.newaxis]

	#Get the background
	background = img - img1

	#Change all pixels in the background that are not black to white
	background[np.where((background > [0,0,0]).all(axis = 2))] = [255,255,255]

	#Add the background and the image
	final = background + img1
	return final


def is_canny_match(img, temp):
    mean_temp_int = np.mean(temp)
    temp_canny = cv2.Canny(temp, int(0.75*mean_temp_int), int(1.75*mean_temp_int))
    mean_img_int = np.mean(img)
    img_canny = cv2.Canny(img, int(0.75*mean_temp_int), int(1.75*mean_temp_int))
    #plt.imshow(img_canny, cmap = "gray")
    #plt.show()
    #plt.imshow(temp_canny, cmap = "gray")
    #plt.show()
    try:
        res = cv2.matchTemplate(img_canny, temp_canny, cv2.TM_CCORR_NORMED)
        #print(res)
        loc = np.where(res > 0.5)
        #print(loc)
        #plt.imshow(res, cmap = "gray")
        #plt.show()
        #print(loc)
        return len(loc[0]) > 0
    except:
        return False
    #print("canny matching points: ", loc)  


def is_diff_optimized(img_patch, template_new):
	difference = cv2.subtract(get_gausian(template_new), get_gausian(img_patch))
	diff = cv2.countNonZero(difference)
	if diff < 100:
		return True
	return False

PATH_TEMPLATE = "Q1/small.png"

for i in range(1, 14):
	PATH_IMG = "Q1/pos_" + str(i) +".jpg"
	img = cv2.imread(PATH_IMG)
	img_show = img.copy()
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
	img_gray = get_gausian(img_gray)
	img_gray = get_laplacian(img_gray)

	template = cv2.imread(PATH_TEMPLATE)
	template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY);
	template = get_laplacian(template)

	scale_percent = 50
	count_subsampled = 0  
	for scale in range(scale_percent, scale_percent + 1):
		img_copy =  img_gray.copy()
		template_new = resize(template, scale)
		res = cv2.matchTemplate(img_copy, template_new, cv2.TM_CCORR_NORMED) 
		w, h = template_new.shape[::-1] 
		threshold = 0.633
		loc = np.where( res >= threshold)  
		l_len = len(loc[0])
		for pt in zip(*loc[::-1]): 
			img_patch = img_copy[pt[1]:pt[1]+(h), pt[0]:pt[0]+(w)]
			if is_canny_match(img_patch, template_new):
				if is_diff_optimized(img_patch, template_new):
					count_subsampled += 1
					cv2.rectangle(img_show, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
		print(str(count_subsampled) + " Sub sampled count for " + PATH_IMG)
		cv2.imshow('image', img_show)
		#cv2.imwrite("pos_" + str(i) +".jpg", img_show)
		cv2.waitKey(0)


