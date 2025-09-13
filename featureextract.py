"""
This is featureextract.py file.

This file is used to get feature from the signer video from 
RWTH-PHOENIX-Weather dataset.
The feature is the sum of magnitude from the signer's hands.

The magnitude is calculated by RLOF dense optical flow.

The hand and body landmarks / points from the signer is taken from mediapipe.
The mediapipe usage on this source code is adapted from:

Mediapipe documentation can be found here:
https://github.com/google-ai-edge/mediapipe/tree/master/docs/solutions

usage:
    python featureextract.py --path <path to video directory> 
                             --groundtruth <path ground truth file>
                             --smooth
                             --figureplot
arguments:
    --path (required):
        Set to the path of RWTH-PHOENIX-Weather video directory.
        This script only accept one video.

    --groundtruth (optional):
        Set to the path of ground truth file.
        It uses RWTH-PHOENIX-Weather automatic annotation file.

    --smooth (optional)
        Set if smooth signal is needed. The smoothing process uses
        Savitzky-Golay filter.

    --figureplot (optional)
        Set if a figure of signal ploting is needed.

author: naserjawas
date: 25 November 2024
"""

import os
# ✅ Limit CPU threads BEFORE importing mediapipe or tensorflow
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

# Optional: make NumPy respect the same thread limit
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import glob
import json
import argparse
import cv2 as cv
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loadgroundtruth import load_gt
from scipy import signal

def load_rwth_phoenix(datapath):
    dirname = str(datapath) + os.sep
    filenames = sorted(glob.glob(dirname + "*-0.png"))
    images = [cv.imread(filename, cv.IMREAD_COLOR)
              for filename in filenames]

    return images

def detect_hands(body, img):
    hands = body.Hands(static_image_mode=True)
    result = hands.process(img)

    if result.multi_hand_landmarks:
        return result
    else:
        return None

def detect_face(body, img):
    face = body.FaceMesh(static_image_mode=True, refine_landmarks=True)
    result = face.process(img)

    if result.multi_face_landmarks:
        return result
    else:
        return None

def detect_pose(body, img):
    pose = body.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True)
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
    for i in range(15, 23):
        y = int(pose.pose_landmarks.landmark[i].y * ih)
        x = int(pose.pose_landmarks.landmark[i].x * iw)
        y = check_range(y, ih)
        x = check_range(x, iw)
        smag += mag[y][x]
        if i % 2 == 0:
            smag_r += mag[y][x]
        else:
            smag_l += mag[y][x]

    return smag, smag_r, smag_l

def make_signal(sig, sflag):
    x = np.array(sig)
    x = x.reshape(1, -1)[0]
    if sflag:
        x_sig = signal.savgol_filter(x, 25, 5)
    else:
        x_sig = x

    return x_sig

def parse_args():
    description = "Program to produce hand, body, and face landmarks"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-p", "--path", help="path to video dir",
                        dest="path", type=str, required=True)
    parser.add_argument("-g", "--groundtruth", help="path to groundtruth file",
                        dest="gtpath", type=str, required=False)
    parser.add_argument("-s", "--smooth", help="set smooth signal",
                        dest="smooth", action="store_true")
    parser.add_argument("-f", "--figureplot", help="set to show figure plot",
                        dest="figureplot", action="store_true")
    parser.add_argument("-d", "--outputdir", help="path to output directory",
                        dest="outputdir", type=str, required=False)

    return parser.parse_args()


def main():
    args = parse_args()
    datapath = Path(args.path)
    images = load_rwth_phoenix(datapath)
    print(f"Load {len(images)} images... OK")

    videoname = datapath.parts[-2]
    print(f"video: {videoname}")

    if args.outputdir is not None:
        outpath = Path(args.outputdir)
        if not outpath.exists():
            print(f"ouputdir: {outpath.resolve()} does not exists")
            exit()

    if args.gtpath is not None:
        gtdata = load_gt(args.gtpath, videoname)
        print(f"Load {len(gtdata)} groundtruth data... OK")
    else:
        gtdata = []
        print(f"No groundtruth data")

    # create mediapipe 
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose

    # create RLOF
    of = cv.optflow.createOptFlow_DenseRLOF()

    summag = []
    summag_r = []
    summag_l = []
    fid = -1
    for image in images:
        fid += 1
        image2 = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        next_i = images[fid].copy()
        if fid == 0:
            prev_i = images[fid].copy()
        else:
            prev_i = images[fid-1].copy()

        # hands = detect_hands(mp_hands, image2)
        # if hands is not None:
        #     for hand_landmarks in hands.multi_hand_landmarks:
        #         image = draw_landmark(image, hand_landmarks.landmark)

        # face = detect_face(mp_face, image2)
        # if face is not None:
        #     for face_mesh in face.multi_face_landmarks:
        #         image = draw_landmark(image, face_mesh.landmark)

        mag, ang = calc_optical_flow(prev_i, next_i, of)

        pose = detect_pose(mp_pose, image2)

        if pose is not None:
            image = draw_pose(image, pose)
            smag, smag_r, smag_l = get_mag_on_pose(mag, pose)
        else:
            smag, smag_r, smag_l = 0, 0, 0

        summag.append(smag)
        summag_r.append(smag_r)
        summag_l.append(smag_l)

        # cv.imshow("image", image)
        # cv.imshow("mag", mag)
        # k = cv.waitKey(0) & 0xFF
        # if k == ord('q'):
        #     cv.destroyAllWindows()
        #     break

    x_summag = make_signal(summag, args.smooth)
    x_summag_r = make_signal(summag_r, args.smooth)
    x_summag_l = make_signal(summag_l, args.smooth)
    x_frmnum = [gt[0] for gt in gtdata]
    x_gtdata = [gt[1] for gt in gtdata]

    # create figure
    px = list(range(len(summag)))
    fig, ax = plt.subplots()
    ax.plot(px, x_summag)
    ax.plot(px, x_summag_r)
    ax.plot(px, x_summag_l)
    for frm, gt in gtdata:
        if gt == 1:
            plt.axvline(x=frm, color="0.8", linestyle=":")

    if args.figureplot:
        plt.show()
    else:
        print(f"x_summag: {x_summag}")
        print(f"x_summag_r: {x_summag_r}")
        print(f"x_summag_l: {x_summag_l}")
        print(f"x_gtdata: {x_gtdata}")

    if args.outputdir is not None:
        filename = str(outpath) + os.sep + videoname
        filenamejson = filename + ".json"
        filenamejpeg = filename + ".jpeg"
        print(f"Saving to {filenamejson}")
        data = {
                "summag": x_summag.tolist(),
                "summag_r": x_summag_r.tolist(),
                "summag_l": x_summag_l.tolist(),
                "frmnum": x_frmnum,
                "gtdata": x_gtdata
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Saving to {filenamejpeg}")
        plt.savefig(filenamejpeg)

if __name__ == "__main__":
    main()


