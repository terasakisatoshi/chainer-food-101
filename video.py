import argparse
import os
import time

import chainer
import chainer.links as L
import chainer.functions as F
import cv2
import numpy as np

from networks.mobilenetv2 import MobilenetV2
from networks.vgg16 import VGG16
from networks.resnet50 import ResNet50
from predict import prepare_setting


def video(args):
    classes = np.genfromtxt(os.path.join(args.dataset, "meta", "classes.txt"),
                            str,
                            delimiter="\n")
    model, preprocess, xp, _ = prepare_setting(args)

    cap = cv2.VideoCapture(0)
    if cap.isOpened() is False:
        print("Error opening video stream or file")
    fps_time = 0
    with chainer.using_config('train', False):
        while cap.isOpened():
            ret_val, img = cap.read()
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_image = cv2.resize(img, (224, 224))
            input_image = input_image.transpose(2, 0, 1)
            input_image = preprocess(input_image).astype(np.float32)
            start = time.time()
            h = model.predictor(xp.expand_dims(xp.array(input_image), axis=0))
            prediction = F.softmax(h)
            if args.device >= 0:
                prediction = xp.asnumpy(prediction[0].data)
            else:
                prediction = prediction[0].data
            top_ten = np.argsort(-prediction)[:10]
            end = time.time()
            print("Elapsed", end - start)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            blank = np.zeros((224, 448, 3)).astype(img.dtype)
            for rank, label_idx in enumerate(top_ten):
                score = prediction[label_idx]
                label = classes[label_idx]
                print('{:>3d} {:>6.2f}% {}'.format(
                    rank + 1, score * 100, label))
                cv2.putText(blank, '{:>3d} {:>6.2f}% {}'.format(
                    rank + 1, prediction[label_idx] * 100, classes[label_idx]),
                    (10, 20 * (rank + 2)),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(blank, "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            title = "Food-101"
            cv2.imshow(title, cv2.hconcat([img, blank]))
            fps_time = time.time()
            """Hit esc key"""
            if cv2.waitKey(1) == 27:
                break


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path", type=str, help="path/to/snapshot e.g. pretrained/model_epoch_100.npz")
    parser.add_argument("dataset", type=str, help="path/to/food-101")
    parser.add_argument("--device", type=int, default=-1,
                        help="specify GPU_ID. If negative, use CPU")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_argument()
    video(args)
