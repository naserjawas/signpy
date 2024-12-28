"""
This is getlandmarks.py file.

This file is used to get the hand and body landmarks / points from the signer.
It uses mediapipe and this source code is adapted from:

Mediapipe documentation can be found here:
https://github.com/google-ai-edge/mediapipe/tree/master/docs/solutions

author: naserjawas
date: 25 November 2024
"""

import os
import glob
import argparse
import cv2 as cv
import mediapipe as mp
from pathlib import Path

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

def draw_landmark(image, landmark):
    ih, iw, _ = image.shape
    for l in landmark:
        y = int(l.y * ih)
        x = int(l.x * iw)
        cv.circle(image, (x, y), 1, (255, 255, 255), -1)

    return image

def draw_pose(image, pose):
    ih, iw, _ = image.shape
    for i in range(25):
        y = int(pose.pose_landmarks.landmark[i].y * ih)
        x = int(pose.pose_landmarks.landmark[i].x * iw)
        cv.circle(image, (x, y), 1, (255, 255, 255), -1)

    return image

def calc_optical_flow(prev_i, next_i, of):
    flow = of.calc(prev_i, next_i, None)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

    return mag, ang

def get_mag_on_pose(mag, pose):
    ih, iw = mag.shape
    summag = 0
    summag_r = 0
    summag_l = 0
    for i in range(15, 23):
        y = int(pose.pose_landmarks.landmark[i].y * ih)
        x = int(pose.pose_landmarks.landmark[i].x * iw)
        summag += mag[y][x]
        if i % 2 == 0:
            summag_r += mag[y][x]
        else:
            summag_l += mag[y][x]

    return summag, summag_r, summag_l

def parse_args():
    description = "Program to produce hand, body, and face landmarks"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-p", "--path", help="path to video dir",
                        dest="path", type=str, required=True)

    return parser.parse_args()

def main():
    args = parse_args()
    datapath = Path(args.path)
    images = load_rwth_phoenix(datapath)
    print(f"Load {len(images)} images... OK")

    # create mediapipe 
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose

    # create RLOF
    of = cv.optflow.createOptFlow_DenseRLOF()

    summag = []
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
            print(f"summag: {get_mag_on_pose(mag, pose)}")


        cv.imshow("image", image)
        cv.imshow("mag", mag)
        k = cv.waitKey(0) & 0xFF
        if k == ord('q'):
            cv.destroyAllWindows()
            break

if __name__ == "__main__":
    main()


