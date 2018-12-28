import cv2
import numpy as np
from random import shuffle
print(cv2.__version__)

def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)


def get_keypoints(img):
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    img_kp = cv2.drawKeypoints(gray, kp, None, color = (128, 0, 128))
    return img_kp, des, kp

def match_using_knn(des_1, des_2, k = 2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    rawMatches = bf.knnMatch(des_1, des_2, 2)
    good = []
    for m,n in rawMatches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    task1_matches_knn = cv2.drawMatches(mountain_1, kp_1 ,mountain_2, kp_2, good, None, flags=2)
    return task1_matches_knn, good

mountain_1 = cv2.imread('mountain1.jpg')
mountain_2 = cv2.imread('mountain2.jpg')

task1_sift1, des_1, kp_1 = get_keypoints(mountain_1)
task1_sift2, des_2, kp_2 = get_keypoints(mountain_2)

show(task1_sift1)
show(task1_sift2)

task1_matches_knn, good = match_using_knn(des_1, des_2, 2)
show(task1_matches_knn)

src_pts = np.float32([ kp_1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp_2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()

print(M)

inliers = []

for x in range(len(matchesMask)):
    if matchesMask[x] == 1:
        inliers.append(x)

shuffle(inliers)

if len(inliers) > 10:
    inliers = inliers[0:10]


task1_matches_inliers = cv2.drawMatches(mountain_1, kp_1 ,mountain_2, kp_2, [good[inlier] for inlier in inliers], None, flags=2)
# show(task1_matches_inliers)

stitcher = cv2.createStitcher(False)
status, result = stitcher.stitch((mountain_1, mountain_2))

print(result)
show(result[1])

# h,w,l = mountain_1.shape
# pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
# dst = cv2.perspectiveTransform(pts, M)
#
# print(len(matchesMask))
# print(len(src_pts))
# print(len(dst_pts))
# matches = []
# ratio=0.75
# # loop over the raw matches
# for m in rawMatches:
# 	# ensure the distance is within a certain ratio of each
# 	# other (i.e. Lowe's ratio test)
# 	if len(m) == 2 and m[0].distance < m[1].distance * ratio:
# 		matches.append((m[0].trainIdx, m[0].queryIdx))
#
#
#
# computing a homography requires at least 4 matches
# if len(matches) > 4:
# 	# construct the two sets of points
# 	ptsA = np.float32([kp_1[i] for (_, i) in matches])
# 	ptsB = np.float32([kp_2[i] for (i, _) in matches])
#
# 	# compute the homography between the two sets of points
# 	(M, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)


# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
#

# img3 = cv2.drawMatches(mountain_1, kp_1, mountain_2, kp_2, good, None, flags=2)
# # show(img3)


# M = [[ 5.88377483e-01,  1.36513235e-01,  2.58515926e+02],
#  [-2.85142602e-01,  8.66037958e-01,  5.15061528e+01],
#  [-7.43481745e-04, -9.07061414e-05,  1.00000000e+00]]
#
# M = np.asarray(M)
# result = cv2.warpPerspective(mountain_1, M,(mountain_1.shape[1] + mountain_2.shape[1], mountain_1.shape[0]))
# result[0:mountain_2.shape[0], 0:mountain_2.shape[1]] = mountain_2
# show(result)
#
# # im_out_1 = cv2.warpPerspective(mountain_1, M, (mountain_1.shape[1] + mountain_2.shape[1],mountain_1.shape[0]))
# #
# # im_out_1[0:mountain_2.shape[0], 0:mountain_2.shape[1]] = mountain_2
# #
# # show(im_out_1)
