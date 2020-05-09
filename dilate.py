import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import os
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

class EyeCoordinates:
  def __init__(self, ex, ey,ew,eh):
    self.ex = ex
    self.ey = ey
    self.ew = ew
    self.eh = eh

def findMax(vector):
    if(len(vector)>=1):
        max=vector[0]
        for i in vector:
            if max<i:
                max=i
        return max

cap = cv2.VideoCapture("can4.mp4") #input video
success,image = cap.read(1)
count = 0
leftEyeArr = []
rightEyeArr = []
imgs=[] #array that keeps images captured from the video
while success:
    if(image is not None):
        #Save captured images to a file with order
        cv2.imwrite("C:/Users/asus/Desktop/pupilDilation/images/frame-%d.jpg" % count, image)
        imgs.append(image) #Also, add that image to the imgs array
        success,image = cap.read()
        count += 1 #Count is for keeping image's order
i=0
for img in imgs: #For all images that we captured
    eyePointsArr=[] #array for keeping eye's location points
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #turn image to grayscale
    eyes = eye_cascade.detectMultiScale(img_gray) #detect all eyes in the image and add it to eyes array

    for (ex, ey, ew, eh) in eyes:
        temp=EyeCoordinates(ex,ey,ew,eh) #keep koordinates as an object
        if(175<ew<250):
            eyePointsArr.append(temp) #if width is in the desired range, add it to array.
    if(len(eyePointsArr)>1): #Again we make sure there are more than one eye.
        if(eyePointsArr[0].ex<eyePointsArr[1].ex): #Detect which eye is the left eye by checking x coordinate
            #If x value of first element is smaller than second element,
            # that means first element is on the left. Assign it to the left eye
            roi_left= img[eyePointsArr[0].ey:(eyePointsArr[0].ey + eyePointsArr[0].eh), eyePointsArr[0].ex:(eyePointsArr[0].ex + eyePointsArr[0].ew)]
            roi_right = img[eyePointsArr[1].ey:(eyePointsArr[1].ey + eyePointsArr[1].eh),
                   eyePointsArr[1].ex:(eyePointsArr[1].ex + eyePointsArr[1].ew)]
        else:
            # If x value of first element is bigger than second element,
            # that means first element is on the right. Assign it to the right eye
            roi_right=img[eyePointsArr[0].ey:(eyePointsArr[0].ey + eyePointsArr[0].eh), eyePointsArr[0].ex:(eyePointsArr[0].ex + eyePointsArr[0].ew)]
            roi_left = img[eyePointsArr[1].ey:(eyePointsArr[1].ey + eyePointsArr[1].eh),
                    eyePointsArr[1].ex:(eyePointsArr[1].ex + eyePointsArr[1].ew)]
    if (len(eyePointsArr) > 1):
        #Threshold values can be editted in:  _, threshold_left = cv2.threshold(gray_roi_left, 30, 255, cv2.THRESH_BINARY_INV)
        gray_roi_left = cv2.cvtColor(roi_left, cv2.COLOR_BGR2GRAY) #turn image to grayscale
        gray_roi_left = cv2.GaussianBlur(gray_roi_left, (7, 7), 0)  # used gaussian method to reduce salt and pepper noise
        _, threshold_left = cv2.threshold(gray_roi_left, 17, 255, cv2.THRESH_BINARY_INV) #apply threshold

        gray_roi_right = cv2.cvtColor(roi_right, cv2.COLOR_BGR2GRAY) #turn image to grayscale
        gray_roi_right = cv2.GaussianBlur(gray_roi_right, (7, 7), 0)  # used gaussian method to reduce salt and pepper noise
        _, threshold_right = cv2.threshold(gray_roi_right, 17, 255, cv2.THRESH_BINARY_INV) #apply threshold

    #apply countours to highlight thresholded areas
        contours, hierarchy = cv2.findContours(threshold_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas=[]
        # for every area catched, add them to an array,
        #finds area of the regions and calculates diameter,
        #adds them to the eye array
        for cnt in contours:
            area = cv2.contourArea(cnt)
            approx=cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt, True), True)
            if 400< area < 2000 :
                area = math.sqrt(area / math.pi)
                areas.append(area)
    #Take biggest area as the pupil
        leftEyeArr.append(findMax(areas))
        i=i+1
        areas.clear()
        contours, hierarchy = cv2.findContours(threshold_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        for cnt in contours: #for every area catched, add them to an array
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if 400 < area < 2000 :
                area=math.sqrt(area/math.pi)
                areas.append(area)
    # Take biggest area as the pupil
        rightEyeArr.append(findMax(areas))

        areas.clear()

#Plot features
fig, axs = plt.subplots(2)
axs[0].set_ylim([0,50])
axs[1].set_ylim([0,50])
axs[0].set_xlabel("images")
axs[0].set_ylabel("areas")
axs[1].set_xlabel("images")
axs[1].set_ylabel("areas")
axs[0].plot(leftEyeArr, color='green',markersize=12)
axs[0].set_title('Left Eye')
axs[1].set_title('Right Eye')
axs[1].plot(rightEyeArr, color='blue')
fig.tight_layout(pad=3.0)

plt.show()
