import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import math

#Load Image
img2 = cv2.imread('car.jpg',1)
cv2.namedWindow("Original Image",cv2.WINDOW_NORMAL)
cv2.imshow("Original Image",img2)

#Gray Color Image
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("Gray Color Image",cv2.WINDOW_NORMAL)
cv2.imshow("Gray Color Image",gray2)

#Noise reduction
bfilter2 = cv2.bilateralFilter(gray2, 11, 17, 17)
#Edge detection
edged2 = cv2.Canny(bfilter2, 30, 200) 
cv2.namedWindow("Edged",cv2.WINDOW_NORMAL)
cv2.imshow("Edged",edged2)

#Find Contours
keypoints2 = cv2.findContours(edged2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours2 = imutils.grab_contours(keypoints2)
contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours2:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
    
#Apply Mask    
mask2 = np.zeros(gray2.shape, np.uint8)
new_image2 = cv2.drawContours(mask2, [location], 0,255, -1)
new_image2 = cv2.bitwise_and(img2, img2, mask=mask2)

new_image3 = cv2.cvtColor(new_image2, cv2.COLOR_BGR2RGB)
cv2.namedWindow("New Image",cv2.WINDOW_NORMAL)
cv2.imshow("New Image",new_image3)

(x,y) = np.where(mask2==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image2 = gray2[x1:x2+1, y1:y2+1]

cropped_image3 = cv2.cvtColor(cropped_image2, cv2.COLOR_BGR2RGB)
cv2.namedWindow("Cropped Image",cv2.WINDOW_NORMAL)
cv2.imshow("Cropped Image",cropped_image3)

#Use OCR
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image3)
print(result)


cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()