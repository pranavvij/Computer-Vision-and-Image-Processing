import math
import cv2
import numpy as np

img = cv2.imread("task2.jpg", 0)
a = np.asarray(img).tolist()

def create_octave_matrix():
	a = math.pow(2, -0.5)
	mult = math.pow(2, 0.5)
	
	OCTAVE_MATRIX = [[0]*5]
	OCTAVE_MATRIX[0][0] = a

	for i in range(1, 4):
		OCTAVE_MATRIX.append([0]*5)
		OCTAVE_MATRIX[i][0] = OCTAVE_MATRIX[i - 1][0] * 2
	
	for i in range(0, 4):
		for j in range(1, 5):
			OCTAVE_MATRIX[i][j] = OCTAVE_MATRIX[i][j - 1] * mult

	return OCTAVE_MATRIX


def get_gausian_xy(x, y, sigma):
	sigma_2 = 2.0*sigma*sigma
	return (math.exp(-(x*x + y*y)/sigma_2))


def gausian_matrix(n,sigma):
	A = [[0] * n]
	s = 0 
	for i in range(0, n - 1):A.append([0] * n)
	for i in range(0, n):
		for j in range(0, n):
			g = get_gausian_xy(i - (n - 1)/2, j - (n - 1)/2, sigma)
			s += g
			A[i][j] = g
	for i in range(0, n):
		for j in range(0, n):
			A[i][j] /= s
	return A

def add_padding(a, n, row_len, col_len):
	MATRIX_1 = []
	padding_len = int((n - 1)/2) 
	for i in range(0, padding_len):
		MATRIX_1.append([0]*(col_len + n - 1))
	for i in range(0, row_len):
		MATRIX_1.append([0]*padding_len + a[i] + [0]*padding_len)
	for i in range(0, padding_len):
		MATRIX_1.append([0]*(col_len + + n - 1))

	return MATRIX_1

def matrix_mult_rect_scan(A, B, n):# WHERE A IS BIGGER THAN B
	padding = int((n - 1)/2)
	row_a_size = len(A)
	col_a_size = len(A[0]) 
	MATRIX_SCANNED = []
	for i in range(0,row_a_size):
		MATRIX_SCANNED.append([0]*col_a_size)

	for i in range(padding, row_a_size - padding):
		for j in range(padding, col_a_size - padding):
			for k in range(0, n):
				for l in range(0, n):
					MATRIX_SCANNED[i][j] += B[k][l]*A[i - padding + k][j - padding + l]
	return MATRIX_SCANNED

def max_min_matrix(A):
	row_a_size = len(A)
	col_a_size = len(A[0]) 
	max_x = 0
	min_x = A[0][0]
	for i in range(0, row_a_size):
		max_x = max(max(A[i]), max_x)
		min_x = min(min(A[i]), min_x)

	return max_x, min_x

def normalize_matrix(A):
	row_a_size = len(A)
	col_a_size = len(A[0]) 
	max_x = 0
	min_x = A[0][0]
	for i in range(0, row_a_size):
		max_x = max(max(A[i]), max_x)
		min_x = min(min(A[i]), min_x)

	for i in range(0, row_a_size):
		for j in range(0, col_a_size):
			A[i][j] = (max_x - A[i][j] )/(max_x - min_x)
	return A

def subtract_matrix(A, B):
	row_a_size = len(A)
	col_a_size = len(A[0]) 
	row_b_size = len(B)
	col_b_size = len(B[0])
	if col_b_size != col_a_size or row_a_size != row_b_size:
		print("error found")
		return 
	for i in range(0, row_b_size):
		for j in range(0, col_b_size):
			A[i][j] = A[i][j] - B[i][j]
	return A
	

def show_image(matrix):
	img2 = np.array(matrix)
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.imshow('image', img2)
	cv2.waitKey(0)


def gausian_matrix_sigma(sigma, MATRIX, n):
	row_len = len(MATRIX)
	col_len = len(MATRIX[0])
	GAUSIAN_MATRIX = gausian_matrix(n, sigma)
	MATRIX  = add_padding(MATRIX, n, row_len, col_len)
	MATRIX_AFTER_MULT = matrix_mult_rect_scan(MATRIX, GAUSIAN_MATRIX, n)
	MATRIX_AFTER_MULT = normalize_matrix(MATRIX_AFTER_MULT)
	return MATRIX_AFTER_MULT


n = 7 # CONSIDERING n to b odd
row_len = len(a)
col_len = len(a[0])
OCTAVE_MATRIX = create_octave_matrix()

for i in range(0, len(OCTAVE_MATRIX)):
	for j in range(0, len(OCTAVE_MATRIX[0]) - 1):
		matrix = gausian_matrix_sigma(OCTAVE_MATRIX[i][j], a, n)
		matrix_1 = gausian_matrix_sigma(OCTAVE_MATRIX[i][j+1], a, n)
		MATRIX_AFTER_MULT_1 = normalize_matrix(subtract_matrix(matrix_1, matrix))
		show_image(MATRIX_AFTER_MULT_1)
		max_x, min_x  = max_min_matrix(MATRIX_AFTER_MULT_1)
		print(max_x, min_x)







