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

def detect(body, img):
    with body.Hands(
        static_image_mode = True,
        max_num_hands = 2,
        min_detection_confidence=0.5
    ) as hands:
        img2 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        result = hands.process(img2)

        if result.multi_hand_landmarks:
            return result
        else:
            return None


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
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    for image in images:
        hands = detect(mp_hands, image)
        if hands is not None:
            for hand_landmarks in hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        cv.imshow("image", image)
        k = cv.waitKey(0) & 0xFF
        if k == ord('q'):
            cv.destroyAllWindows()
            break

if __name__ == "__main__":
    main()


