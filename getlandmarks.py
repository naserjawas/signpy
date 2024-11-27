"""
This is getlandmarks.py file.

This file is used to get the hand and body landmarks / points from the signer.
It uses mediapipe and this source code is adapted from:
https://mediapipe.readthedocs.io/en/latest/solutions/hands.html

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

    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose

    for image in images:
        image2 = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # hands = detect_hands(mp_hands, image2)
        # if hands is not None:
        #     for hand_landmarks in hands.multi_hand_landmarks:
        #         image = draw_landmark(image, hand_landmarks.landmark)

        # face = detect_face(mp_face, image2)
        # if face is not None:
        #     for face_mesh in face.multi_face_landmarks:
        #         image = draw_landmark(image, face_mesh.landmark)

        pose = detect_pose(mp_pose, image2)
        if pose is not None:
            image = draw_pose(image, pose)

        cv.imshow("image", image)
        k = cv.waitKey(0) & 0xFF
        if k == ord('q'):
            cv.destroyAllWindows()
            break

if __name__ == "__main__":
    main()


