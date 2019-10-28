import argparse
import numpy as np
from sklearn.cluster import KMeans
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
    parts = filename.split(".")

    model = models.load_model('resnet50_coco_best_v2.1.0.h5', backbone_name='resnet50')

    cap = cv2.VideoCapture(filename)

    camera_cut = CameraCut()

    team_hists = np.array([[0.00200573715878, 0.008174931463, 0.01083576065,
                            0.010703199637, 0.005602396121, 0.003398788058,
                            0.003046975742, 0.002051472299, 0.001472585509,
                            0.0004638160026],
                           [0.000803887693714, 0.006247766973138,
                            0.008450147937586, 0.01134667238931,
                            0.004603156519655, 0.003117572623929,
                            0.003535995531379, 0.003621568075172,
                            0.00310307416, 0.001896991398214]])

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
            out = cv2.VideoWriter('{}_{}.mp4'.format(parts[0], cut_counter), fourcc, 50.0,
                      (int(cap.get(3)), int(cap.get(4))))
            cut_counter += 1
            counter = 0
        draw = np.array(frame)

        if counter % 5 == 0:
            # Every 5 frames use retinanet to find players in the frame
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            processed_frame = preprocess_image(processed_frame)
            processed_frame, scale = resize_image(processed_frame)

            boxes, scores, _ = model.predict_on_batch(np.expand_dims(processed_frame, axis=0))
            boxes /= scale
            players = []
            confident_boxes = []
            teams = []
            for box, score in zip(boxes[0], scores[0]):
                if score < 0.5:
                    break
                box = box.astype(int)
                player_image = draw[box[1]:box[3], box[0]:box[2], :]
                confident_boxes.append(box)
                player_image = cv2.cvtColor(player_image, cv2.COLOR_RGB2GRAY)
                player_hist = np.histogram(player_image, density=True)[0]
                dist = np.sum(np.power((team_hists - player_hist), 2), axis=1)
                min_index = np.argmin(dist)
                teams.append([0, 255, 0])
                if min_index == 0 and dist[0] < 2e-05:
                    teams[-1] = [255, 0, 0]
                elif dist[1] < 2e-05:
                    teams[-1] = [0, 0, 255]

        # Plot the most recent detections on the current frame
        for box, team in zip(confident_boxes, teams):
            draw_box(draw, box, color=team)

        # Write the current frame with annotations to the VideoWriter
        # import matplotlib.pyplot as plt
        # plt.imshow(draw)
        # plt.show()
        out.write(draw)
        counter += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_analysis(args.filename)
