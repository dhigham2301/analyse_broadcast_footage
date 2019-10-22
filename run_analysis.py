import argparse
import numpy as np
import cv2

import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color


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


class CameraCut:
    """
    This class declares a naive approach to determine camera cuts in the
    footage. It creates histograms of the red and blue channels and if there is
    a large difference in the histograms over subsequent frames it is declared
    a camera cut. This fails when there are transition graphics.
    """

    def __init__(self):
        self.red_hist = None
        self.blue_hist = None

    def new_frame(self, frame):
        """
        Determine if a new frame is a camera cut.

        Parameters
        ---------
        frame: Image
            The new frame

        Returns
        -------
        cut: boolean
            Whether the new frame is a cut from the previous frame
        """
        cut = True
        new_red_hist = np.histogram(frame[:, :, 2])[0]
        new_blue_hist = np.histogram(frame[:, :, 0])[0]
        if self.red_hist is not None:
            red_diff = np.sum(np.abs(self.red_hist - new_red_hist))
            blue_diff = np.sum(np.abs(self.blue_hist - new_blue_hist))
            if red_diff + blue_diff < 1000000:
                cut = False
        self.red_hist = new_red_hist
        self.blue_hist = new_blue_hist
        return cut


def run_analysis(filename):
    """
    Run analysis on the provided file

    Parameters
    ---------
    filename: String
        The wall placement strategies the user has provided
    """
    model = models.load_model('resnet50_coco_best_v2.1.0.h5', backbone_name='resnet50')

    cap = cv2.VideoCapture(filename)

    camera_cut = CameraCut()

    cut_counter = 1
    counter = 0
    while(cap.isOpened()):
        print(counter)
        ret, frame = cap.read()
        if camera_cut.new_frame(frame):
            # If this frame is a camera cut, lose the existing VideoWriter and
            # open a new one
            if cut_counter > 1:
                out.release()
            fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
            out = cv2.VideoWriter('{}.mp4'.format(cut_counter), fourcc, 50.0,
                      (int(cap.get(3)), int(cap.get(4))))
            cut_counter += 1
            counter = 0
        draw = frame.copy()

        if counter % 5 == 0:
            # Every 5 frames use retinanet to find players in the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = preprocess_image(frame)
            frame, scale = resize_image(frame)

            boxes, scores, labels = model.predict_on_batch(np.expand_dims(frame, axis=0))
            boxes /= scale

        # Plot the most recent detections on the current frame
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < 0.5:
                break

            color = label_color(label)

            b = box.astype(int)
            draw_box(draw, b, color=color)

            caption = "{:.3f}".format(score)
            draw_caption(draw, b, caption)

        # Write the current frame with annotations to the VideoWriter
        out.write(draw)
        counter += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_analysis(args.filename)
