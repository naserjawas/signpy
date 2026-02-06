"""
This is support.py file.

This file contains supporting functions.

author: naserjawas
date: 30 December 2024
"""

import os
import glob
import cv2 as cv
import numpy as np
from scipy import signal

def load_gt(filename, videofile):
    with open(filename, "r") as f:
        lines = f.readlines()
    f.close()

    gtdata = []
    for line in lines:
        frame, sign = line.split(" ")
        signid = int(sign)
        framedata = frame.split(os.sep)

        # for the selected videofile
        if framedata[-3] == videofile:
            fn = framedata[-1].split("_")[-1]
            fnid = int(fn[2:8])
            # sign id for "si" is 3693
            gtdata.append((fnid, signid))

    return gtdata

def load_rwth_phoenix(datapath):
    dirname = str(datapath) + os.sep
    filenames = sorted(glob.glob(dirname + "*-0.png"))
    images = [cv.imread(filename, cv.IMREAD_COLOR)
              for filename in filenames]

    return images

def detect_hands(hands, img):
    result = hands.process(img)

    if result.multi_hand_landmarks:
        return result
    else:
        return None

def detect_face(face, img):
    result = face.process(img)

    if result.multi_face_landmarks:
        return result
    else:
        return None

def detect_pose(pose, img):
    result = pose.process(img)

    if result.pose_landmarks:
        return result
    else:
        return None

def check_range(orig, maxm):
    newvalue = 0
    if orig < 0:
        newvalue = 0
    elif orig >= maxm:
        newvalue = maxm-1
    else:
        newvalue = orig

    return newvalue

def draw_landmark(image, landmark):
    ih, iw, _ = image.shape
    for l in landmark:
        y = int(l.y * ih)
        x = int(l.x * iw)
        y = check_range(y, ih)
        x = check_range(x, iw)
        cv.circle(image, (x, y), 1, (255, 255, 255), -1)

    return image

def draw_pose(image, pose):
    ih, iw, _ = image.shape
    for i in range(25):
        y = int(pose.pose_landmarks.landmark[i].y * ih)
        x = int(pose.pose_landmarks.landmark[i].x * iw)
        y = check_range(y, ih)
        x = check_range(x, iw)
        cv.circle(image, (x, y), 1, (255, 255, 255), -1)

    return image

def calc_optical_flow(prev_i, next_i, of):
    flow = of.calc(prev_i, next_i, None)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

    return mag, ang

def get_mag_on_pose(mag, pose):
    ih, iw = mag.shape
    smag = 0
    smag_r = 0
    smag_l = 0
    pose_r = []
    pose_l = []
    for i in range(15, 23):
        y = int(pose.pose_landmarks.landmark[i].y * ih)
        x = int(pose.pose_landmarks.landmark[i].x * iw)
        y = check_range(y, ih)
        x = check_range(x, iw)
        smag += mag[y][x]
        if i % 2 == 0:
            smag_r += mag[y][x]
            pose_r.append((x,y))
        else:
            smag_l += mag[y][x]
            pose_l.append((x,y))

    return smag, smag_r, smag_l, pose_r, pose_l

def make_signal(sig, sflag):
    x = np.array(sig)
    x = x.reshape(1, -1)[0]
    if sflag:
        x_sig = signal.savgol_filter(x, 25, 5)
    else:
        x_sig = x

    return x_sig

def pose_to_numpy(pose):
    x = np.array([
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in pose.pose_landmarks.landmark
                 ], dtype=np.float32)
    return x
