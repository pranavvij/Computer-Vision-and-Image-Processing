import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def get_white_percentage(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        if color[0] > 240 and color[1] > 240 and color[2] > 240:
        	return percent


def image_color_percentage(img):
	img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
	clt = KMeans(n_clusters=10) #cluster number
	clt.fit(img)
	hist = find_histogram(clt)
	percent = get_white_percentage(hist, clt.cluster_centers_)
	return percent

img = cv2.imread("small.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(image_color_percentage(img))