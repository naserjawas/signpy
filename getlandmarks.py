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
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This files produces hand body landmarks from the signer.")
    parser.add_argument("-p", "--path", help="path to video dir", dest="path", required=True)

    args = parser.parse_args()

    datapath = Path(args.path)
    dirname =  str(datapath) + os.sep
    print("dirname:", dirname)
    filenames = sorted(glob.glob(dirname + "*-0.png"))
    images = [cv.imread(filename, cv.IMREAD_COLOR)
              for filename in filenames]

    print("Load", len(images), "images... OK")

