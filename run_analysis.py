import argparse
import numpy as np
import cv2

def build_parser():
    parser = argparse.ArgumentParser(
        description="Perform an analyse on broadcast footage")

    parser.add_argument(
        "-f", "--filename",
        help=("The filename of the broadcast footage to be analysed"),
        dest="filename",
        required=True
    )

    return parser


def run_analysis(filename):
    """
    Run analysis on the provided file

    Parameters
    ---------
    filename: String
        The wall placement strategies the user has provided
    """
    cap = cv2.VideoCapture('filename')

    while(cap.isOpened()):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    args = build_parser().parse_args()
    run_analysis(args.filename)
