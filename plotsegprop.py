"""
This is plotsegprop.py file.

This file is used to read saved segmentation proposal and plot it to a graph.

usage:
    python plotsegprop.py --featurefile <path to npz file>

argument:
    --featurefile (required):
        Set to the npz feature file path.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    description = "Program to load segmentation proposal and plot to a graph."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-f", "--segpropfile", help="npz segprop file path",
                        dest="segpropfile", type=str, required=True)

    return parser.parse_args()

def plotgroundtruth(plt, gtboundaries):
    for i, gt in enumerate(gtboundaries):
        # print(i, gt, gtsign[i])
        if gt == 1:
            plt.axvline(x=i, color="0.8", linestyle=":")
        if gt == 2:
            plt.axvline(x=i, color="0.2", linestyle=":")

def main():
    args = parse_args()

    filename = args.segpropfile
    print("Load " ,filename, "... OK")
    data  = np.load(filename)

    boundary_score = data['boundary_score']
    speed_score = data['speed_score']
    direction_score = data['direction_score']
    orientation_score = data['orientation_score']
    peaks = data['peaks']
    threshold = data['threshold']
    gtboundaries = data['gtboundaries']
    gtsign = data['gtsign']
    gtgloss = data['gtgloss']
    gtmanual = data['gtmanual']
    videoname = data['videoname']

    # time axis
    T = len(boundary_score)
    t = np.arange(T)

    # plot 1
    plt.figure()
    plt.plot(t, boundary_score, label="boundary_score")
    plotgroundtruth(plt, data['gtboundaries'])
    plt.title("boundary_score")
    plt.xlabel("Frame")
    plt.ylabel("boundary_score")
    # plot 2
    plt.figure()
    plt.plot(t, speed_score, label="speed_score")
    plotgroundtruth(plt, data['gtboundaries'])
    plt.title("speed_score")
    plt.xlabel("Frame")
    plt.ylabel("speed_score")
    # plot 3
    plt.figure()
    plt.plot(t, direction_score, label="direction_score")
    plotgroundtruth(plt, data['gtboundaries'])
    plt.title("direction_score")
    plt.xlabel("Frame")
    plt.ylabel("direction_score")
    # plot 4
    plt.figure()
    plt.plot(t, orientation_score, label="orientation_score")
    plotgroundtruth(plt, data['gtboundaries'])
    plt.title("orientation_score")
    plt.xlabel("Frame")
    plt.ylabel("orientation_score")

    plt.show()

if __name__ == "__main__":
    main()
