import cv2
from cv2 import DIST_L12
import numpy as np

img = cv2.imread('./img/pills2.jpg')    
gray = cv2.imread('./img/pills2.jpg', cv2.IMREAD_GRAYSCALE)

kernel = np.ones((11,11),np.uint8)
inrange = cv2.inRange(img, (0, 120, 200), (0, 255, 255))
inrange = cv2.dilate(inrange, kernel, iterations = 1)
masked = cv2.bitwise_and(img, img.copy(), mask=inrange)


dist = cv2.distanceTransform(inrange, cv2.DIST_L2, 5)
dist_ = cv2.normalize(dist, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

_, th1 = cv2.threshold(dist_, 145, 255, cv2.THRESH_BINARY)
_, th2 = cv2.threshold(dist_, 175, 255, cv2.THRESH_BINARY_INV)
dist = cv2.bitwise_and(th1, th2)
dist = cv2.dilate(dist, kernel, iterations = 1)

dist2 = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
dist2 = cv2.normalize(dist2, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
_, dist2 = cv2.threshold(dist2, 155, 255, cv2.THRESH_BINARY)
dist2 = cv2.dilate(dist2, kernel, iterations = 1)

dist = cv2.subtract(dist, dist2)

kernel = np.ones((5, 5),np.uint8)
dist = cv2.erode(dist, kernel, iterations = 3)
dist = cv2.dilate(dist, kernel, iterations = 1)
kernel = np.ones((3, 3),np.uint8)
dist = cv2.erode(dist, kernel, iterations = 1)
dist = cv2.bitwise_or(dist, dist2)

################################## contour ##################################################
contours, hierarchy = cv2.findContours(dist, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(img, contours, -1, (0, 0, 255), 2, hierarchy=hierarchy, maxLevel=1)

unit = inrange[300:400, 200:320]
unit_contours, unit_hierarchy = cv2.findContours(unit, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
rect = cv2.minAreaRect(unit_contours[0])
box = cv2.boxPoints(rect)
unit_box = np.int0(box)
unit_box_w, unit_box_h = unit_box.max(axis=0) - unit_box.min(axis=0)


count = 0
for contour in contours:
    contour_ = contour.squeeze()
    if len(contour_.shape) >= 2:
        x_min, y_min = contour_.min(axis=0)
        x_max, y_max = contour_.max(axis=0)


        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        min_x, min_y = box.min(axis=0)
        max_x, max_y = box.max(axis=0)
        square = (max_x - min_x) ** 2 + (max_y - min_y) ** 2
        if square <= 50:
            continue
        
        count += 1
        middle_x, middle_y = ((box.max(axis=0) + box.min(axis=0)) / 2).astype(np.int0)
        x_min = middle_x - int(unit_box_w/2)
        y_min = middle_y - int(unit_box_h/2)
        x_max = middle_x + int(unit_box_w/2)
        y_max = middle_y + int(unit_box_h/2)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 56, 255), 2)
        # cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
        
        # img_ = img.copy()
        # cv2.putText(img_, 'Count: %d' %count, (30, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
        # cv2.imshow('img', img_)
        # cv2.waitKey(500)

cv2.putText(img, 'Count: %d' %count, (30, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)



cv2.imshow('img', img)
cv2.imshow('dist', dist)
# cv2.imshow('dist2', dist2)
# cv2.imshow('canny', canny)
# cv2.imshow('masked', masked)
# cv2.imshow('inrange', inrange)
# cv2.imshow('dilation', dilation)
# cv2.imshow('contour', white_background)
# cv2.imshow('contour2', white_background2)
cv2.waitKey(0)
cv2.destroyAllWindows()



# 흑백 처리
# gray, threshold, inrange

# 이미지 처리
# blur, edge, morphology

# 테두리 좌표
# contour, hough