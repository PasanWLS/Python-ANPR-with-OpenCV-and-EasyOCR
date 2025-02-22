import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import math

#Load Image
img = cv2.imread('car.jpeg',1)
cv2.namedWindow("Original Image",cv2.WINDOW_NORMAL)
cv2.imshow("Original Image",img)

#Rotate Image
h, w = img.shape[0], img.shape[1]
rotation_mat = cv2.getRotationMatrix2D((h/2,w/2),-8,0.75)
rotated_img = cv2.warpAffine(img,rotation_mat,(h,w))
cv2.namedWindow("Rotated Image",cv2.WINDOW_NORMAL)
cv2.imshow("Rotated Image",rotated_img)

#Gray Color Image
gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("Gray Color Image",cv2.WINDOW_NORMAL)
cv2.imshow("Gray Color Image",gray)

#Noise reduction
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
#Edge detection
edged = cv2.Canny(bfilter, 30, 200) 
cv2.namedWindow("Edged",cv2.WINDOW_NORMAL)
cv2.imshow("Edged",edged)

#Find Contours and Apply Mask
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
    
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(rotated_img, rotated_img, mask=mask)

new_image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
cv2.namedWindow("New Image",cv2.WINDOW_NORMAL)
cv2.imshow("New Image",new_image1)

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

cropped_image1 = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
cv2.namedWindow("Cropped Image",cv2.WINDOW_NORMAL)
cv2.imshow("Cropped Image",cropped_image1)

#Use OCR
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image1)

print(result)


cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()