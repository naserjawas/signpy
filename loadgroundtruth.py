"""
This is loadgroundtruth.py file.

This file is used to load the annotation file and change its form
into a segmentation ground truth.

author: naserjawas
date: 30 December 2024
"""

import os

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
            # convert to the segment ground truth
            # 0: sign
            # 1: silent/"si" (non-sign)
            # sign id for "si" is 3693
            if signid == 3693:
                gtdata.append([fnid, 1])
            else:
                gtdata.append([fnid, 0])

    return gtdata
