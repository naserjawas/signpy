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

def parse_args():
    description = "Program to produce hand, body, and face landmarks"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-p", "--path", help="path to video dir",
                        dest="path", type=str, required=True)

    return parser.parse_args()

def main():
    args = parse_args()
    datapath = Path(args.path)
    dirname = str(datapath) + os.sep
    filenames = sorted(glob.glob(dirname + "*-0.png"))
    images = [cv.imread(filename, cv.IMREAD_COLOR)
              for filename in filenames]

    print(f"Load {len(images)} images... OK")

if __name__ == "__main__":
    main()


