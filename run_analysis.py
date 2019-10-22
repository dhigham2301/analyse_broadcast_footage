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

    counter = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if counter % 5 == 0:
            draw = frame.copy()

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = preprocess_image(frame)
            frame, scale = resize_image(frame)

            boxes, scores, labels = model.predict_on_batch(np.expand_dims(frame, axis=0))
            boxes /= scale

            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # scores are sorted so we can break
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < 0.5:
                    break

                color = label_color(label)

                b = box.astype(int)
                draw_box(draw, b, color=color)

                caption = "{:.3f}".format(score)
                draw_caption(draw, b, caption)

            cv2.imshow('frame', draw)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        counter += 1
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    args = build_parser().parse_args()
    run_analysis(args.filename)
