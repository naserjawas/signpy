"""
This is landmarkextract.py file.

This file is used to get landmark from a video file.
The landmark is extracted using mediapipe.

usage:
    python landmarkextract.py --path <path to video>
                              --outputdir <path to result directory>

arguments:
    --path (required):
        Set to the video path.

    --outputdir (optional):
        Set to the path of output directory.
"""

import os
# Limit CPU threads BEFORE importing mediapipe or tensorflow
# os.environ["OMP_NUM_THREADS"] = "8"                                                                    │~
# os.environ["TF_NUM_INTRAOP_THREADS"] = "8"                                                             │~
# os.environ["TF_NUM_INTEROP_THREADS"] = "8"                                                             │~

# Optional: make NumPy respect the same thread limit                                                   │~
# os.environ["MKL_NUM_THREADS"] = "8"                                                                    │~
# os.environ["NUMEXPR_NUM_THREADS"] = "8"

import argparse
import cv2 as cv
import mediapipe as mp
import numpy as np
import glob

from pathlib import Path

def parse_args():
    description = "Program to extract human pose, face, hand landmarks from a video"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-p", "--path", help="path to video input",
                        dest="path", type=str, required=True)
    parser.add_argument("-d", "--outputdir", help="path to output directory",
                        dest="outputdir", type=str, required=False)
    return parser.parse_args()

def landmarks_to_array(landmarks, num_points):
    if landmarks is None:
        return np.zeros((num_points, 3))
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])

def normalise_pose(pose):
    left_hip = pose[23]
    right_hip = pose[24]
    left_shoulder = pose[11]
    right_shoulder = pose[12]

    hip_centre = (left_hip + right_hip) / 2
    shoulder_centre = (left_shoulder + right_shoulder) / 2

    scale = np.linalg.norm(shoulder_centre - hip_centre) + 1e-6

    pose_norm = (pose - hip_centre) / scale
    return pose_norm

def normalise_hand(hand, wrist):
    return hand - wrist

def compute_velocity(sequence):
    velocity = np.diff(sequence, axis=0)
    zero_frame = np.zeros_like(sequence[:1])
    velocity = np.concatenate([zero_frame, velocity], axis=0)
    return velocity


def main():
    args = parse_args()
    datapath = Path(args.path)

    dirname = str(datapath) + os.sep
    filenames = sorted(glob.glob(dirname + "*-0.png"))
    images = [cv.imread(filename, cv.IMREAD_COLOR_RGB)
              for filename in filenames]
    print(f"Load {len(images)} images... OK")

    videoname = datapath.parts[-2]
    print(f"video: {videoname}")

    if args.outputdir is not None:
        outpath = Path(args.outputdir)
        if not outpath.exists():
            print(f"outputdir: {outpath.resolve()} does not exists")
            exit()

    mp_holistic = mp.solutions.holistic

    pose_seq = []
    face_seq = []
    left_hand_seq = []
    right_hand_seq = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        refine_face_landmarks=True
    ) as holistic:
        for image in images:
            rgb = image.copy()
            results = holistic.process(rgb)

            pose = landmarks_to_array(results.pose_landmarks, 33)
            face = landmarks_to_array(results.face_landmarks, 468)
            left_hand = landmarks_to_array(results.left_hand_landmarks, 21)
            right_hand = landmarks_to_array(results.right_hand_landmarks, 21)

            pose_norm = normalise_pose(pose)

            left_hand_norm = normalise_hand(left_hand, pose[15])
            right_hand_norm = normalise_hand(right_hand, pose[16])

            pose_seq.append(pose_norm)
            face_seq.append(face)
            left_hand_seq.append(left_hand_norm)
            right_hand_seq.append(right_hand_norm)

    pose_seq = np.array(pose_seq)
    face_seq = np.array(face_seq)
    left_hand_seq = np.array(left_hand_seq)
    right_hand_seq = np.array(right_hand_seq)

    velocity_pose = compute_velocity(pose_seq)
    velocity_hands = compute_velocity(
        np.concatenate([left_hand_seq, right_hand_seq], axis=1)
    )

    if args.outputdir is not None:
        filename = str(outpath) + os.sep + videoname
    else:
        filename = videoname
    output_path = filename + "_features.npz"

    np.savez_compressed(
        output_path,
        pose=pose_seq,
        face=face_seq,
        left_hand=left_hand_seq,
        right_hand=right_hand_seq,
        velocity_pose=velocity_pose,
        velocity_hands=velocity_hands
    )
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
