import cv2
import numpy as np
from collections import deque
import argparse
import imutils
import sys
import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
args = vars(ap.parse_args())

cap = cv2.VideoCapture('varejao02.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=10, detectShadows=False)
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=10, nmixtures=5, backgoundRatio=0.7, noiseSigma=0)


ball_coord = deque(maxlen = args["buffer"])

while (1):

    ret, frame = cap.read()
    #'''Tirei só pq não tenho a boblioteca e o video está em 369x640'''
    #frame2 = cv2.resize(frame, width=600)
    frame2 = frame
    if frame is None:
        #sai do loop quando acabar o vídeo
        break;
    fgmask = fgbg.apply(frame2)


    # filtros(peciso melhorar muito)
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    #frame3 = cv2.Canny(gray, 100, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(fgmask, kernel, iterations=3)
    mask2 = cv2.dilate(mask, kernel, iterations=9)


    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Filter by Convexity

    params.filterByArea = True
    params.minArea = 300
    params.maxArea = 1500  # The dot in 20pt font has area of about 30
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
    
    if len(keypoints) > 0 :
        for kpts in keypoints:
            ball_coord.appendleft(kpts.pt)
            cv2.circle(frame2, (int(kpts.pt[0]),int(kpts.pt[1])), 5, (0, 0, 255), -1)


    # Show blobs
    cv2.imshow("Keypoints", im_with_keypoints)



    #cv2.imshow('frame', frame2)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

ball_coord = np.array( ball_coord )

plt.scatter(ball_coord[:, 0], ball_coord[:,1 ])
plt.show()
