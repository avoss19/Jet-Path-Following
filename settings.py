import numpy as np
import logging as log
import sys

MAX_THROTTLE = 8

MOTOR = 0
SERVO = 1

MIDPOINT = 90

# v4l2-ctl -d /dev/video0
# v4l2-ctl --list-formats-ext
# SOURCE = "media/IMG_2186.mov"
# SOURCE = "recording_demo.avi"
# SOURCE = "/home/voss/Projects/JetFume/outside1.avi"
SOURCE = 0
X_RES = 640
Y_RES = 360
#X_RES = 1920
#Y_RES = 1080
FPS = 60
RECORD = False
RECORD_FILE = "video%d.avi"

CUDA = False

THRESH = 20
MAX_LINE_GAP = 20
MIN_LINE_LENGTH = 10
MIN_LINE_THETA = .35 # rad

LINE_RANK = .3

IMG_SCALE = 1

# ROI = np.float32([[0, Y_RES], [(X_RES / 10) * 2, (Y_RES * .4)], [X_RES - (X_RES / 10)*2, Y_RES * .4], [X_RES, Y_RES]])

# ROI_DEMO = np.float32([[0, Y_RES * .8], [(X_RES / 10) * 1, (Y_RES * .5)], [X_RES - (X_RES / 10)*1, Y_RES * .5], [X_RES, Y_RES * .8]])
ROI = np.float32([[X_RES * .2, Y_RES * .9], [(X_RES / 10) * 4, (Y_RES * .5)], [X_RES - (X_RES / 10)*4, Y_RES * .5], [X_RES * .8, Y_RES * .9]])

# ROI = np.float32([[0, Y_RES], [(X_RES / 10) * 2, Y_RES / 2], [X_RES - (X_RES / 10)*2, Y_RES / 2], [X_RES, Y_RES]])
RES = np.float32([[0, Y_RES], [0, 0], [X_RES, 0], [X_RES, Y_RES]])

#
# LOWER_MASK_COLOR = np.array([225, 0, 105], dtype="uint8")
# UPPER_MASK_COLOR = np.array([260, 30, 140], dtype = "uint8")

LOWER_MASK_COLOR = np.array([0, 120, 0], dtype="uint8")
UPPER_MASK_COLOR = np.array([20, 180, 20], dtype = "uint8")

# Camera Calibration
IMG = 0.409024301145824
MTX = np.array([[1.39047450e+03, 0.00000000e+00, 9.56345020e+02],
 [0.00000000e+00, 1.39537130e+03, 5.85222941e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
DIST = np.array([[-0.34292298 , 0.13932197,  0.00220346, -0.0009022,  -0.05778542]])

# Config Logging
log.basicConfig(
    filename='demo.log',
    encoding='utf-8',
    level=log.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
    )
log.getLogger().addHandler(log.StreamHandler(sys.stdout))
