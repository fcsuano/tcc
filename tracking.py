import cv2
import numpy as np
from collections import deque
import argparse
import imutils
import sys



src1 = cv2.imread("canvas.png");
src2 = cv2.imread("canvas2.png");
dst = new cv2.Mat();
mask = new cv2.Mat();
dtype = -1;
cv.subtract(src1, src2, dst, mask, dtype);
src1.delete(); src2.delete(); dst.delete(); mask.delete();