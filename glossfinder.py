"""
This is glossfinder.py file.

This file is used to show list of glosses in a video and find matching glosses
elsewhere in other videos.

usage:
    python glossfinder.py --v_id <video id> --onlymain
"""
import argparse

def parse_args():
    description = "Program to show list of glosses and find matching glosses"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-v", "--v_id", help="video id",
                        dest="v_id", type=int, required=True)
    parser.add_argument("-m", "--onlymain", help="flag to show only main transcript",
                        dest="onlymain", action='store_true')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    v_id = args.v_id
    onlymain = args.onlymain

    # load manual ground truth
    gtmanual = []
    gtvideo = []
    manualfile = "../dataset/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/train.corpus.csv"
    with open(manualfile, "r") as mf:
        next(mf)
        for line in mf:
            data = line.split("|")

            videoname = data[0]
            gtvideo.append(videoname)

            sentence = data[-1]
            glosses = sentence.split(" ")
            gtmanual.append(glosses)

    mainvideo = gtvideo[v_id]
    target = gtmanual[v_id]
    print(f"Main video: {mainvideo}")
    print(f"Transcript: {target}")

    if not onlymain:
        for t in target:
            print(f"target gloss: {t}")
            e = 0
            for gt in gtmanual:
                if t in gt:
                    e += 1
                    print(f"{e}. t:{t}, {gt}")

