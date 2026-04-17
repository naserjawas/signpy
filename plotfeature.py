"""
This is plotfeature.py file.

This file is used to read saved features and plot it to a graph.

usage:
    python plotfeature.py --featurefile <path to npz file>

argument:
    --featurefile (required):
        Set to the npz feature file path.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    description = "Program to load feature and plot to a graph."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-f", "--featurefile", help="npz feature file path",
                        dest="featurefile", type=str, required=True)

    return parser.parse_args()

def main():
    args = parse_args()

    filename = args.featurefile
    print("Load " ,filename, "... OK")
    data  = np.load(filename)

    velocity_hands = data["velocity_hands"]                 # (T, 42, 3)
    direction_angle_hands = data["direction_angle_hands"]   # (T, 42)
    left_orient_change = data["left_orient_change"]         # (T,)
    right_orient_change = data["right_orient_change"]       # (T,)
    left_orientation = data["left_orientation"]             # (T,)
    right_orientation = data["right_orientation"]           # (T,)

    # wrist indices
    left_wrist_velocity = velocity_hands[:, 0,: ]
    right_wrist_velocity = velocity_hands[:, 21,: ]
    # wrist magnitude
    left_wrist_speed = np.linalg.norm(left_wrist_velocity, axis=1)
    right_wrist_speed = np.linalg.norm(right_wrist_velocity, axis=1)
    # direction change
    direction_change_mean = np.mean(direction_angle_hands, axis=1)
    # orientation magnitude
    left_mag = np.linalg.norm(left_orientation, axis=1)
    right_mag = np.linalg.norm(right_orientation, axis=1)
    # time axis
    T = len(left_wrist_speed)
    t = np.arange(T)

    # plot 1
    plt.figure()
    plt.plot(t, left_wrist_speed, label="Left Wrist Velocity")
    plt.plot(t, right_wrist_speed, label="Right Wrist Velocity")
    plt.legend()
    plt.title("Wrist Velocity Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Speed")
    # plot 2
    plt.figure()
    plt.plot(t, direction_change_mean)
    plt.title("Direction Change (Angle)")
    plt.xlabel("Frame")
    plt.ylabel("Radians")
    # plot 3
    plt.figure()
    plt.plot(t, left_orient_change, label="Left Orientation Change")
    plt.plot(t, right_orient_change, label="Right Orientation Change")
    plt.legend()
    plt.title("Orientation Change")
    plt.xlabel("Frame")
    plt.ylabel("Radians")
    # plot 4
    plt.figure()
    plt.plot(t, left_mag, label="Left Orientation Magnitude")
    plt.plot(t, right_mag, label="Right Orientation Magnitude")
    plt.legend()
    plt.title("Orientation Magnitude")
    plt.xlabel("Frame")
    plt.ylabel("Magnitude")

    plt.show()

if __name__ == "__main__":
    main()
