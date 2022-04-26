import cv2
import numpy as np

img = cv2.imread('./img/pills.jpeg')
img = img[:582,:]
gray = cv2.imread('./img/pills.jpeg', cv2.IMREAD_GRAYSCALE)
canny = cv2.Canny(gray, 100, 255)

kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(canny, kernel, iterations = 1)
dilation = cv2.dilate(canny, kernel, iterations = 1)

contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

white_background = np.full(img.shape, 255, dtype=np.uint8)
white_background2 = np.full(img.shape, 255, dtype=np.uint8)
# cv2.drawContours(white_background, contours, -1, (0, 0, 255), 2, hierarchy=hierarchy, maxLevel=1)
# cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

count = 0
min_max = []
for contour in contours:
    if len(contour) >= 10:
        contour_ = contour.squeeze()
        x_min, y_min = contour_.min(axis=0)
        x_max, y_max = contour_.max(axis=0)
        
        if x_max - x_min >= 30 and y_max - y_min >= 30:
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 56, 255), 2)
            count += 1

            
cv2.putText(img, 'Count: %d' %count, (30, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)


cv2.imshow('img', img)
# cv2.imshow('canny', canny)
# cv2.imshow('dilation', dilation)
# cv2.imshow('contour', white_background)
# cv2.imshow('contour2', white_background2)
cv2.waitKey(0)
cv2.destroyAllWindows()



# 흑백 처리
# gray, threshold

# 이미지 처리
# blur, edge, morphology

# 테두리 좌표
# contour, hough