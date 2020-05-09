import numpy as np
import cv2

#This file can be run after dilate.py and images folder is created by dilate.py
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
img_dir="images/frame-108.jpg" #Specify the image created by dilate.py
img=cv2.imread(img_dir)
img_copy=cv2.imread(img_dir)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

eyes = eye_cascade.detectMultiScale(gray)
i=0
for (ex, ey, ew, eh) in eyes:
    i=i+1
    #print(ex,ey,ew,eh)
    roi_left = img[ey:(ey + eh), ex:(ex + ew)]
    cv2.imshow('roi_left circles', roi_left)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gray_roi_left = cv2.cvtColor(roi_left, cv2.COLOR_BGR2GRAY)  # turn image to grayscale
    cv2.imshow('COLOR_BGR2GRAY circles', gray_roi_left)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    gray_roi_left = cv2.GaussianBlur(gray_roi_left, (7, 7), 0)  # used gaussian method to reduce salt and pepper noise
    cv2.imshow('GaussianBlur circles', gray_roi_left)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    _, threshold_left = cv2.threshold(gray_roi_left, 17, 255, cv2.THRESH_BINARY_INV)  # apply thresh
    cv2.imshow('threshold_left circles', threshold_left)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

