import cv2
import numpy as np
from collections import deque
import argparse
import sys
import math


ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=30,
	help="max buffer size")
args = vars(ap.parse_args())

pts = deque(maxlen=args["buffer"])
cap = cv2.VideoCapture('varejao02.mp4')

success,image = cap.read()
fgbg = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=200, detectShadows=False)

def is_empty(any_structure):
    if any_structure:
        return False
    else:
        return True

x_past=0
y_past=0
checa_erro=0
x=0
y=0
conta_frames=0
count = 0


while (1):

    ret, frame = cap.read()
    bgmask = fgbg.apply(frame)

    kernel = np.ones((2,2),np.uint8)
    blur = cv2.GaussianBlur(bgmask, (3, 3), 0)
    mask = cv2.erode(blur, kernel, iterations=1)
    bola = cv2.dilate(mask, kernel, iterations=6)

    mask2 = cv2.dilate(blur, kernel, iterations=15)
    mask3 = cv2.erode(mask2, kernel, iterations= 20)
    pessoa = cv2.dilate(mask3, kernel,iterations= 5)

    mask4 = cv2.subtract(bola,pessoa)
    mask5=cv2.dilate(mask4, kernel, iterations=5)
    mask6=cv2.GaussianBlur(mask5, (1, 1), 10)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Filter by Convexity
    params.filterByArea = True
    params.minArea = 400
    params.maxArea = 1100
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.filterByInertia = True
    params.minInertiaRatio = 0.8

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(np.invert(mask6))

    if not (is_empty(keypoints)):
        x = keypoints[0].pt[0]
        y = keypoints[0].pt[1]

        if math.sqrt((x_past-x)**2+(y_past-y)**2)<65:
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

    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (255, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    for i in np.arange(1, len(pts)):
        cv2.line(im_with_keypoints, pts[i - 1], pts[i], (255, 255, 255), 2)

    cv2.imshow("Keypoints",im_with_keypoints)

    #cv2.imwrite('frame_'+str(checa_erro)+'.png', im_with_keypoints)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
out.release()








