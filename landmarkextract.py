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
# os.environ["OMP_NUM_THREADS"] = "8"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
# os.environ["TF_NUM_INTEROP_THREADS"] = "8"

# Optional: make NumPy respect the same thread limit
# os.environ["MKL_NUM_THREADS"] = "8"
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
    # velocity = np.diff(sequence, axis=0)
    # zero_frame = np.zeros_like(sequence[:1])
    # velocity = np.concatenate([zero_frame, velocity], axis=0)
    velocity = np.diff(sequence, axis=0, prepend=sequence[0:1])
    return velocity

def compute_direction(velocity):
    norm = np.linalg.norm(velocity, axis=-1, keepdims=True) + 1e-6
    direction = velocity / norm
    return direction

def compute_direction_change(direction):
    delta_dir = np.diff(direction, axis=0, prepend=direction[0:1])
    return delta_dir

def compute_direction_angle(velocity):
    v1 = velocity[:-1]
    v2 = velocity[1:]

    dot = np.sum(v1 * v2, axis=-1)
    norm1 = np.linalg.norm(v1, axis=-1)
    norm2 = np.linalg.norm(v2, axis=-1)

    cos_theta = dot / (norm1 * norm2 + 1e-6)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle = np.arccos(cos_theta)

    angle = np.concatenate([np.zeros_like(angle[:1]), angle], axis=0)
    return angle

def compute_hand_orientation(hand):
    wrist = hand[:, 0, :]
    index = hand[:, 8, :]

    orientation = index - wrist
    return orientation

def normalise_vector(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-6
    v_norm = v / norm
    return v_norm

def compute_orientation_change(orientation):
    orientation = normalise_vector(orientation)

    v1 = orientation[:-1]
    v2 = orientation[1:]

    dot = np.sum(v1 * v2, axis=-1)
    norm1 = np.linalg.norm(v1, axis=-1)
    norm2 = np.linalg.norm(v2, axis=-1)

    cos_theta = dot / (norm1 * norm2 + 1e-6)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle = np.arccos(cos_theta)

    angle = np.concatenate([np.zeros_like(angle[:1]), angle], axis=0)
    return angle


def main():
    args = parse_args()
    datapath = Path(args.path)

    dirname = str(datapath) + os.sep
    filenames = sorted(glob.glob(dirname + "*-0.png"))
    images = [cv.imread(filename, cv.IMREAD_COLOR)
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
    mp_drawing = mp.solutions.drawing_utils

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
            rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
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

    direction_pose = compute_direction(velocity_pose)
    direction_hands = compute_direction(velocity_hands)

    direction_angle_pose = compute_direction_angle(velocity_pose)
    direction_angle_hands = compute_direction_angle(velocity_hands)

    left_orientation = compute_hand_orientation(left_hand_seq)
    right_orientation = compute_hand_orientation(right_hand_seq)

    left_orient_change = compute_orientation_change(left_orientation)
    right_orient_change = compute_orientation_change(right_orientation)

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
        velocity_hands=velocity_hands,

        direction_pose=direction_pose,
        direction_hands=direction_hands,

        direction_angle_pose=direction_angle_pose,
        direction_angle_hands=direction_angle_hands,

        left_orientation=left_orientation,
        right_orientation=right_orientation,

        left_orient_change=left_orient_change,
        right_orient_change=right_orient_change
    )
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
