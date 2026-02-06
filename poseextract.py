"""
This is poseextract.py file.

This file is used to get pose feature from RWTH-PHOENIX-Weather dataset.

The pose landmarks is taken from mediapipe.
The mediapipe usage on this source code is adapted from:

Mediapipe documentation can be found here:
https://github.com/google-ai-edge/mediapipe/tree/master/docs/solutions

usage:
    python poseextract.py --path <path to video directory>
                          --groundtruth <path ground truth file>
                          --outputdir <path to result directory>
arguments:
    --path (required):
        Set to the path of RWTH-PHOENIX-Weather video directory.
        This script only accept one video.

    --groundtruth (optional):
        Set to the path of ground truth file.
        It uses RWTH-PHOENIX-Weather automatic annotation file.

    --groundtruth (optional):
        Set to the path of result directory.

author: naserjawas
date: 05 February 2026
"""

import os
# Limit CPU threads BEFORE importing mediapipe or tensorflow
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"

# Optional: make NumPy respect the same thread limit
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import argparse
import cv2 as cv
import mediapipe as mp
import numpy as np
from pathlib import Path
from support import load_gt, load_rwth_phoenix
from support import detect_hands, detect_face, detect_pose
from support import check_range
from support import draw_pose, draw_landmark
from support import pose_to_numpy

def parse_args():
    description = "Program to produce hand, body, and face landmarks"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-p", "--path", help="path to video dir",
                        dest="path", type=str, required=True)
    parser.add_argument("-g", "--groundtruth", help="path to groundtruth file",
                        dest="gtpath", type=str, required=False)
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

    hand_detector = mp_hands.Hands(static_image_mode=True)
    face_detector = mp_face.FaceMesh(static_image_mode=True, refine_landmarks=True)
    pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True)

    pose_sequence = []
    fid = -1
    for image in images:
        fid += 1
        image2 = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # hands = detect_hands(hand_detector, image2)
        # if hands is not None:
        #     for hand_landmarks in hands.multi_hand_landmarks:
        #         image = draw_landmark(image, hand_landmarks.landmark)

        # face = detect_face(face_detector, image2)
        # if face is not None:
        #     for face_mesh in face.multi_face_landmarks:
        #         image = draw_landmark(image, face_mesh.landmark)

        pose = detect_pose(pose_detector, image2)

        if pose is not None:
            image = draw_pose(image, pose)
            pose_sequence.append(pose_to_numpy(pose))
        else:
            pose_sequence.append(np.zeros((33, 4), dtype=np.float32))

        cv.imshow("image", image)
        k = cv.waitKey(0) & 0xFF
        if k == ord('q'):
            cv.destroyAllWindows()
            break

    x_pose_sequence = np.stack(pose_sequence)
    x_frmnum = [gt[0] for gt in gtdata]
    x_gtdata = [gt[1] for gt in gtdata]

    print(f"x_pose_sequence: {x_pose_sequence}")
    print(f"x_gtdata: {x_gtdata}")

    if args.outputdir is not None:
        filename = str(outpath) + os.sep + videoname
        filenamenpz = filename + "_pose.npz"

        print(f"Saving to {filenamenpz}")
        np.savez(
            filenamenpz,
            x_pose_sequence=x_pose_sequence,
            frmnum=np.array(x_frmnum),
            gtdata=np.array(x_gtdata)
        )

if __name__ == "__main__":
    main()


