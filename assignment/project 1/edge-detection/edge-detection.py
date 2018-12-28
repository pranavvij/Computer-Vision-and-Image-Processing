import cv2
import numpy as np
import copy
img = cv2.imread("task1.png", 0)
a = np.asarray(img).tolist() #### Converting Numpy Image to 2 D array

SOBEL_X = [   
	[1, 2, 1],
	[0, 0, 0],
	[-1,-2,-1]
]

SOBEL_Y = [
	[-1,0, 1],
	[-2,0, 2],
	[-1,0, 1]
]

m = len(a) ## rows
n = len(a[0]) ## columns

SOBEL = 0
for sobel in [SOBEL_X, SOBEL_Y]:
	X = [[0]*(n+2)] 
	for i in range(0, m):
		X.append([0] + a[i] + [0])

	X.append([0]*(n+2))
	Y = copy.deepcopy(X)

	max_x = 0
	min_x = X[1][1]

	for i in range(1, m + 1):
		for j in range(1, n + 1):
			for k in range(-1, 2):
				for l in range(-1, 2):
					try:
						Y[i][j] += (sobel[k + 1][l + 1] * X[i + k][j + l]) 
					except Exception as e:
						print(e)
		max_x = max(max(Y[i]), max_x)
		min_x = min(min(Y[i]), min_x)


	for i in range(1, m + 1):
		for j in range(1, n + 1):
			Y[i][j] = int((max_x - Y[i][j])/(max_x - min_x)*255)

	img2 = np.array(Y)
	if SOBEL == 0:
		cv2.imwrite('SOBEL_X.png',img2)
	else:
		cv2.imwrite('SOBEL_Y.png',img2)
	SOBEL+=1

