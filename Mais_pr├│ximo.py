import cv2
import numpy as np
from collections import deque
import argparse
import imutils
import sys
import math

import cv2

VIDEO = 'video.mp4'

cap = cv2.VideoCapture('/home/francisco/Documentos/TCC/varejao02.mp4')
c = 110
while cap.isOpened():
   res, frame = cap.read()
   if c<200:
    nomeDoFrame2 = 'frame-%d.jpg' % c
    c=c+10
    cv2.imwrite(nomeDoFrame2,frame)

cap.release()