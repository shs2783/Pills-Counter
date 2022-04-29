import cv2
import numpy as np
from scipy.ndimage import label



img = cv2.imread('./img/pills2.jpg')    
gray = cv2.imread('./img/pills2.jpg', cv2.IMREAD_GRAYSCALE)

kernel = np.ones((11,11),np.uint8)
inrange = cv2.inRange(img, (0, 120, 200), (0, 255, 255))  # 알약 색상 추출
inrange = cv2.dilate(inrange, kernel, iterations = 1)  # 모폴로지 연산 (잡티 제거)
# masked = cv2.bitwise_and(img, img.copy(), mask=inrange)





boarder = inrange - cv2.erode(inrange, None)

################################ calculate distance #####################################
dist = cv2.distanceTransform(inrange, cv2.DIST_L2, 5)  # 거리 계산
dist_ = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

# # dist_ = cv2.equalizeHist(dist_)  # 평탄화
# # _, dist_ = cv2.threshold(dist_, 242, 255, cv2.THRESH_BINARY)  # 일정 거리 이상인 부분만 흰색 표시
# # dist = dist_

_, th1 = cv2.threshold(dist_, 145, 255, cv2.THRESH_BINARY)  # 일정 거리 이상인 부분만 흰색 표시
# _, th2 = cv2.threshold(dist_, 175, 255, cv2.THRESH_BINARY_INV)  # 많이 하얀 부분은 검은색 표시 (붙어 있는 오브젝트 처리를 위함)
# dist = cv2.bitwise_and(th1, th2)  # 위 두개 이미지 합치기
# dist = cv2.dilate(dist, kernel, iterations = 1)  # 선 굵게
# dist = th1


# dist = th1
canny = cv2.Canny(gray, 30, 200)
canny = cv2.dilate(canny, None)
canny[canny==255] = 100
th1[th1==255] = 150
dist = canny + th1
dist = cv2.bitwise_or(canny, th1)

img[dist>=151] = (0, 0, 255)

marker, ncc = label(th1)
marker = marker * (255/ncc)
marker[boarder==255] = 255
boarder[dist>=151] = 255
dist = boarder
# dist = th1
# marker = marker.astype(np.int32)
# cv2.watershed(img, marker)
# marker[marker==-1] = 0

# marker = marker.astype(np.uint8)
# marker = 255 - marker
# marker[marker != 255] = 0
# marker = cv2.dilate(marker, None)
# img[marker==255] = (0, 0, 255)




# ret, labels, stats, centroid = cv2.connectedComponentsWithStats(th1, labels=inrange.shape, connectivity=8, ltype=cv2.CV_32S)

# max_val = np.where(labels == 47)
# for i in range(len(max_val[0])):
#     cv2.circle(img, (max_val[1][i], max_val[0][i]), 5, 255, -1)


# for i in range(len(stats)):
#     x, y, w, h, area = stats[i]
#     if 250 <= area <= 10000:
#         hist = cv2.calcHist([dist_[y:y+h, x:x+w]], [0], None, [256], [0, 256])
#         central_x, central_y = int(centroid[i][0]), int(centroid[i][1])
        
#         cv2.rectangle(img, (x, y), (x+w, y+h), 180, 2)
#         cv2.circle(img, (central_x, central_y), 5, 150, -1)
#         cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, 180, 2)


# ########################## process overlapping object #########################################
# dist2 = cv2.distanceTransform(dist, cv2.DIST_L2, 5)  # 다시 한 번 거리 계산
# dist2 = cv2.normalize(dist2, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
# _, dist2 = cv2.threshold(dist2, 155, 255, cv2.THRESH_BINARY)   # 일정 거리 이상인 부분만 흰색 표시
# dist2 = cv2.dilate(dist2, kernel, iterations = 1)

# dist = cv2.subtract(dist, dist2)  # 기존의 흑백 영상과 다른 부분만 표시
# kernel = np.ones((5, 5),np.uint8)  # 모폴로지 연산으로 자잘한 점들 제거
# dist = cv2.erode(dist, kernel, iterations = 3)
# dist = cv2.dilate(dist, kernel, iterations = 1)
# kernel = np.ones((3, 3),np.uint8)
# dist = cv2.erode(dist, kernel, iterations = 1)
# dist = cv2.bitwise_or(dist, dist2)  # 기존 흑백 영상과 합치기

# ##################################### contour ################################################
# contours, hierarchy = cv2.findContours(dist, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# # contours, hierarchy = cv2.findContours(inrange, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# # cv2.drawContours(img, contours, -1, (0, 0, 255), 2, hierarchy=hierarchy, maxLevel=1)

# unit = inrange[300:400, 200:320]
# unit_contours, unit_hierarchy = cv2.findContours(unit, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# # unit_contours, unit_hierarchy = cv2.findContours(unit, 1, 2)
# rect = cv2.minAreaRect(unit_contours[0])
# box = cv2.boxPoints(rect)
# unit_box = np.int0(box)
# unit_box_w, unit_box_h = unit_box.max(axis=0) - unit_box.min(axis=0)


# ################################### draw contour ############################################
# count = 0
# for contour in contours:
#     contour_ = contour.squeeze()
#     if len(contour_.shape) >= 2:
#         x_min, y_min = contour_.min(axis=0)
#         x_max, y_max = contour_.max(axis=0)


#         rect = cv2.minAreaRect(contour)
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)

#         min_x, min_y = box.min(axis=0)
#         max_x, max_y = box.max(axis=0)
#         square = (max_x - min_x) ** 2 + (max_y - min_y) ** 2
#         if square <= 120:
#             continue
        
#         count += 1
#         middle_x, middle_y = ((box.max(axis=0) + box.min(axis=0)) / 2).astype(np.int0)
#         x_min = middle_x - int(unit_box_w/2)
#         y_min = middle_y - int(unit_box_h/2)
#         x_max = middle_x + int(unit_box_w/2)
#         y_max = middle_y + int(unit_box_h/2)
#         cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 56, 255), 2)
#         # cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
        
#         # img_ = img.copy()
#         # cv2.putText(img_, 'Count: %d' %count, (30, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
#         # cv2.imshow('img', img_)
#         # cv2.waitKey(500)

# cv2.putText(img, 'Count: %d' %count, (30, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)



cv2.imshow('dist', dist)
cv2.imshow('canny', canny)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# 흑백 처리
# gray, threshold, inrange

# 이미지 처리
# blur, edge, morphology

# 테두리 좌표
# contour, hough