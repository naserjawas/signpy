"""
This is segmentproposal.py file.

This file is used to calculate segmentation proposal.

usage:
    python segmentproposal.py --featurefile <path to npz file>

argument:
    --featurefile (required):
        Set to the npz feature file path.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
# -----------------------------------------------------------------------------
from support import load_gt
# -----------------------------------------------------------------------------

def parse_args():
    description = "Program to calculate segmentation proposal."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-f", "--featurefile", help="npz feature file path",
                        dest="featurefile", type=str, required=True)
    # -------------------------------------------------------------------------
    parser.add_argument("-g", "--groundtruth", help="path to groundtruth file",
                        dest="gtpath", type=str, required=False)
    # -------------------------------------------------------------------------

    return parser.parse_args()

def normalise(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)

def delta(x):
    return np.abs(np.diff(x, prepend=x[0]))

def main():
    args = parse_args()

    filename = args.featurefile
    print("Load " ,filename, "... OK")
    data  = np.load(filename)

    # -------------------------------------------------------------------------
    videoname = filename.split(os.sep)
    videoname = videoname[-1][:-13]
    if args.gtpath is not None:
        # load data
        gtdata = load_gt(args.gtpath, videoname)
        print(f"Video name: {videoname}")
        print(f"Load {len(gtdata)} groundtruth data... OK")
        # load class label
        gtlabels = {}
        classfile = args.gtpath[:-15]+"trainingClasses.txt"
        with open(classfile, "r") as cf:
            next(cf)
            for line in cf:
                value, key = line.strip().split()
                gtlabels[int(key)] = value
        print(f"Load {len(gtlabels)} gtlabels data... OK")
        # load manual ground truth
        gtmanual = []
        manualfile = args.gtpath[:-25] + "manual/train.corpus.csv"
        with open(manualfile, "r") as mf:
            next(mf)
            for line in mf:
                linedata = line.split('|')
                linevideo = linedata[0].strip()
                if linevideo == videoname:
                    gtmanual = linedata[-1].strip().split()
        print(f"Load {len(gtmanual)} gtmanual data... OK")
        print(gtmanual)
    else:
        gtdata = []
        gtlabels = {}
        gtmanual = []
        print(f"No groundtruth data")

    gtboundaries = []
    gtsign = []
    gtgloss = []
    prvgt = 0
    for frm, gt in gtdata:
        if frm == 0:
            gtboundaries.append(0)
        else:
            if gt not in [prvgt-1, prvgt, prvgt+1, 3693]:
                gtboundaries.append(1)
            elif gt == 3693:
                gtboundaries.append(2)
            else:
                gtboundaries.append(0)
        prvgt = gt
        gtsign.append(gt)
        gtgloss.append(gtlabels[gt])
    print(f"Load {len(gtgloss)} gtgloss data... OK")
    print(gtgloss)
    # -------------------------------------------------------------------------

    velocity_hands = data["velocity_hands"]                 # (T, 42, 3)
    direction_angle_hands = data["direction_angle_hands"]   # (T, 42)
    left_orient_change = data["left_orient_change"]         # (T,)
    right_orient_change = data["right_orient_change"]       # (T,)
    left_orientation = data["left_orientation"]             # (T,)
    right_orientation = data["right_orientation"]           # (T,)

    # wrist indices
    left_wrist_velocity = velocity_hands[:, 0,: ]
    right_wrist_velocity = velocity_hands[:, 21,: ]
    # wrist magnitude / speed
    left_wrist_speed = np.linalg.norm(left_wrist_velocity, axis=1)
    right_wrist_speed = np.linalg.norm(right_wrist_velocity, axis=1)
    # delta magnitude / speed
    left_dv = delta(left_wrist_speed)
    right_dv = delta(right_wrist_speed)
    # hand direction change
    left_direction_angle = direction_angle_hands[:, :21]
    right_direction_angle = direction_angle_hands[:, 21:]
    # direction change
    direction_change_mean = np.mean(direction_angle_hands, axis=1)
    left_direction_change_mean = np.mean(left_direction_angle, axis=1)
    right_direction_change_mean = np.mean(right_direction_angle, axis=1)
    # orientation magnitude
    left_mag = np.linalg.norm(left_orientation, axis=1)
    right_mag = np.linalg.norm(right_orientation, axis=1)
    # time axis
    T = len(left_wrist_speed)
    t = np.arange(T)

    # speed score
    speed_score = np.maximum(left_dv, right_dv)
    # direction score
    direction_score = np.maximum(left_direction_change_mean,
                                 right_direction_change_mean)
    # orientation score
    orientation_score = np.maximum(left_orient_change,
                                   right_orient_change)

    # normalise score
    speed_score = normalise(speed_score)
    direction_score = normalise(direction_score)
    orientation_score = normalise(orientation_score)

    # smooth
    speed_score = savgol_filter(speed_score, 7, 3)
    direction_score = savgol_filter(direction_score, 7, 3)
    orientation_score = savgol_filter(orientation_score, 7, 3)

    # combine
    alpha, beta, gamma = 0.5, 0.2, 0.3
    boundary_score = (alpha * speed_score +
                      beta  * direction_score +
                      gamma * orientation_score)

    # peak detection
    threshhold = np.percentile(boundary_score, 80)
    peaks, _ = find_peaks(boundary_score,
                          height=threshhold,
                          distance=10)

    # # plot
    # plt.figure()
    # plt.plot(t, boundary_score)
    # plt.plot(peaks, boundary_score[peaks], 'x')
    # plt.title("Boundary Score")
    # plt.xlabel("Frame")
    # plt.ylabel("Score")
    # # -------------------------------------------------------------------------
    # for i, gt in enumerate(gtboundaries):
    #     # print(i, gt, gtsign[i])
    #     if gt == 1:
    #         plt.axvline(x=i, color="0.8", linestyle=":")
    #     if gt == 2:
    #         plt.axvline(x=i, color="0.2", linestyle=":")
    # # -------------------------------------------------------------------------
    # plt.show()

if __name__ == "__main__":
    main()
