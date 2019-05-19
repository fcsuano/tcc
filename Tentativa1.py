import cv2
import numpy as np
from collections import deque
import argparse
import imutils
import sys
import math

def is_empty(any_structure)   :
    if any_structure:
        return False
    else:
        return True
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=20,
	help="max buffer size")
args = vars(ap.parse_args())
pts = deque(maxlen=args["buffer"])
cap = cv2.VideoCapture('varejao02.mp4')
#fgbg = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=100, detectShadows=False)
fgbg = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=100, detectShadows=False)
x_past=0
y_past=0
checa_erro=0
x=0
y=0
C=400
while cap.isOpened():

    ret, frame = cap.read()
    frame2 = imutils.resize(frame)
    #fgmask = fgbg.apply(frame2)
    fgmask = fgbg.apply(frame2)



    # filtros(peciso melhorar muito)
    #gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    #frame3 = cv2.Canny(gray, 100, 200)

    kernel = np.ones((2,2),np.uint8)
    blur = cv2.GaussianBlur(fgmask, (3, 3), 0)
    mask = cv2.erode(blur, kernel, iterations=1)
    bola = cv2.dilate(mask, kernel, iterations=10)
    bola2 = cv2.GaussianBlur(bola,(1,1),0)

    mask2 = cv2.dilate(blur, kernel, iterations=10)
    pessoa = cv2.erode(mask2, kernel, iterations= 15)
    pessoa2 = cv2.dilate(pessoa, kernel,iterations= 5)


    mask5 = cv2.subtract(bola,pessoa2)
    #mask5=cv2.dilate(mask5, kernel, iterations=5)

    #mask6 = cv2.add()


    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Filter by Convexity

    params.filterByArea = True
    params.minArea = 300
    params.maxArea = 1000  # The dot in 20pt font has area of about 30
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.8

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(np.invert(mask5))
    if not (is_empty(keypoints)):
        x = keypoints[0].pt[0]
        y = keypoints[0].pt[1]
    if math.sqrt((x_past-x)**2+(y_past-y)**2)<40:
        x_past=x
        y_past=y
        center=(int(x_past),int(y_past))
        pts.appendleft(center)
        checa_erro=0
    if (checa_erro>10) and (not(is_empty(keypoints))):
        del pts
        pts = deque(maxlen=args["buffer"])
        x = keypoints[0].pt[0]
        y = keypoints[0].pt[1]
        x_past=x
        y_past=y
        center=(int(x_past),int (y_past))
        pts.appendleft(center)
        checa_erro=0
    checa_erro=checa_erro+1
    im_with_keypoints = cv2.drawKeypoints(frame2, keypoints, np.array([]), (255, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    for i in np.arange(1, len(pts)):
        cv2.line(im_with_keypoints, pts[i - 1], pts[i], (255, 255, 255), 2)

    cv2.imshow("fgbg",im_with_keypoints)

    Frame400 = 'frame-%d.jpg' % x
    cv2.imwrite(Frame400, frame)

    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()







