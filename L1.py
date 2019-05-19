import cv2
import numpy as np
from collections import deque
import argparse
import imutils
import sys



cap = cv2.VideoCapture('varejao02.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=10, detectShadows=False)
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=10, nmixtures=5, backgoundRatio=0.7, noiseSigma=0)




while (1):

    ret, frame = cap.read()
    frame2 = imutils.resize(frame, width=600)
    fgmask = fgbg.apply(frame2)

 
    # filtros(peciso melhorar muito)
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    #frame3 = cv2.Canny(gray, 100, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(fgmask, kernel, iterations=3)
    mask2 = cv2.dilate(mask, kernel, iterations=9)
    #fgmask = fgbg.apply(mask2)

    #ret1, th1 = cv2.threshold(gray, 170, 255, cv2.THRESH_TRUNC)
    #blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #ret3, th3 = cv2.threshold(blur, 170, 255, cv2.THRESH_OTSU)

    #_, contours, _ = cv2.findContours(mask2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

    #centres = []
    #for i in range(len(contours)):
     #   moments = cv2.moments(contours[i])
      #  if moments['m00'] != 0:

       #     centres.append((int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])))
           # cv2.circle(mask2, centres[-1], 3, (0, 0, 0), -1)



    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Filter by Convexity

    params.filterByArea = True
    params.minArea = 300
    params.maxArea = 1000  # The dot in 20pt font has area of about 30
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.8

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(np.invert(mask2))




    im_with_keypoints = cv2.drawKeypoints(frame2, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



    # Show blobs
    cv2.imshow("Keypoints", im_with_keypoints)



    #cv2.imshow('frame', frame2)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()


