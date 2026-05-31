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
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d

def parse_args():
    description = "Program to load segmentation proposal and plot to a graph."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-f", "--segpropfile", help="npz segprop file path",
                        dest="segpropfile", type=str, required=True)
    parser.add_argument("-s", "--smoothen", help="smoothen the signal",
                        dest="smoothen", action='store_true')

    return parser.parse_args()

def plotgroundtruth(plt, gtboundaries, gtmanual):
    manual_id = 0
    for i, gt in enumerate(gtboundaries):
        # print(i, gt, gtsign[i])
        if gt == 1:
            plt.axvline(x=i, color="red", linestyle=":")
            plt.text(x=i+1, y=0, s=gtmanual[manual_id], color="red",
                     fontsize='x-small', rotation=45)
            manual_id += 1
        if gt == 2:
            plt.axvline(x=i, color="0.8", linestyle=":")

def find_peaks_valleys(signal):
    peaks, _ = find_peaks(signal, prominence=0.3)
    valleys, _ = find_peaks(-signal, prominence=0.3)

    return peaks, valleys

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

    if args.smoothen:
        # boundary_score = gaussian_filter1d(boundary_score, sigma=2)
        # speed_score = gaussian_filter1d(speed_score, sigma=2)
        # direction_score = gaussian_filter1d(direction_score, sigma=2)
        # orientation_score = gaussian_filter1d(orientation_score, sigma=2)
        boundary_score = savgol_filter(boundary_score, window_length=11, polyorder=3)
        speed_score = savgol_filter(speed_score, window_length=11, polyorder=3)
        direction_score = savgol_filter(direction_score, window_length=11, polyorder=3)
        orientation_score = savgol_filter(orientation_score, window_length=11, polyorder=3)

    # plot 1
    plt.figure()
    plt.plot(t, boundary_score, label="boundary_score")
    plotgroundtruth(plt, gtboundaries, gtmanual)
    p, v = find_peaks_valleys(boundary_score)
    plt.plot(p, boundary_score[p], 'rx')
    plt.plot(v, boundary_score[v], 'rx')
    plt.title("boundary_score")
    plt.xlabel("Frame")
    plt.ylabel("boundary_score")
    # plot 2
    plt.figure()
    plt.plot(t, speed_score, label="speed_score")
    plotgroundtruth(plt, gtboundaries, gtmanual)
    p, v = find_peaks_valleys(speed_score)
    plt.plot(p, speed_score[p], 'rx')
    plt.plot(v, speed_score[v], 'rx')
    plt.title("speed_score")
    plt.xlabel("Frame")
    plt.ylabel("speed_score")
    # plot 3
    plt.figure()
    plt.plot(t, direction_score, label="direction_score")
    plotgroundtruth(plt, gtboundaries, gtmanual)
    p, v = find_peaks_valleys(direction_score)
    plt.plot(p, direction_score[p], 'rx')
    plt.plot(v, direction_score[v], 'rx')
    plt.title("direction_score")
    plt.xlabel("Frame")
    plt.ylabel("direction_score")
    # plot 4
    plt.figure()
    plt.plot(t, orientation_score, label="orientation_score")
    plotgroundtruth(plt, gtboundaries, gtmanual)
    p, v = find_peaks_valleys(orientation_score)
    plt.plot(p, orientation_score[p], 'rx')
    plt.plot(v, orientation_score[v], 'rx')
    plt.title("orientation_score")
    plt.xlabel("Frame")
    plt.ylabel("orientation_score")
    # print data
    print(gtmanual)

    plt.show()

if __name__ == "__main__":
    main()
