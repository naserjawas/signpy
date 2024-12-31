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


def main():
    gtfilename = "../../../PhDResearch/workingdir/dataset/RWTHPHOENIXWeather2014/phoenix2014-release/phoenix-2014-multisigner/annotations/automatic/train.alignment"
    videofile = "01April_2010_Thursday_heute_default-0"
    gtdata = load_gt(gtfilename, videofile)
    print(gtdata)


if __name__ == "__main__":
    main()
