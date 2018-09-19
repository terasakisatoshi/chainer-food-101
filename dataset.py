import json
import logging
import os
import random

import chainer
from chainer.datasets import LabeledImageDataset
from chainer.datasets import TransformDataset
from chainercv import transforms

try:
    import cv2
    ENAVLE_CV2 = True
except ImportError:
    ENAVLE_CV2 = False

import numpy as np


CONTRAST_RANGE = np.linspace(0.7, 1.3, 61, endpoint=True)
logger = logging.getLogger(__name__)


BLACKLIST = ["lasagna/3787908",
             "steak/1340977",
             "bread_pudding/1375816"]


def get_pairs(meta_dir, train=True):
    classes = np.genfromtxt(os.path.join(
        meta_dir, "classes.txt"), str, delimiter="\n")
    classes = {label: klass for label, klass in enumerate(classes)}
    json_file = "train.json" if train else "test.json"
    with open(os.path.join(meta_dir, json_file), 'r') as f:
        klass2path = json.load(f)
    pairs = []
    for label, klass in classes.items():
        paths = klass2path[klass]
        for p in paths:
            if p in BLACKLIST:
                logger.info("{} is in BLACKLIST".format(p))
                continue
            pairs.append(
                (os.path.join("food-101", "images", p + ".jpg"), label))
    return pairs


class FoodDataset(chainer.dataset.DatasetMixin):

    def __init__(self, train=True):
        pairs = get_pairs("food-101/meta", train=train)
        self.base = LabeledImageDataset(pairs)
        self.train = True
        self.pairs = pairs

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        imgpath = self.base._pairs[i][0]
        image = image.copy()

        if self.train:
            contrast_value = np.random.choice(CONTRAST_RANGE)
            image = transforms.random_flip(image, x_random=True, y_random=True)
            image = transforms.random_rotate(image, return_param=False)
            image = contrast_value * image
            if image.shape[1] >= 224 and image.shape[2] >= 224:
                image = transforms.random_crop(image, size=(224, 224))
        image = transforms.resize(image, (224, 224))

        if image.shape[0] == 1:
            """
            REMARK: all images are not color images.
            one may find there exists gray scale image
            e.g. food-101/images/steak/1340977.jpg
            Surprisingly, this is also not steak. Just photo of family lol.
            Who made this data :(
            """
            logger.info("gray scale image found ={}".format(imgpath))
            logger.info("We will convert RGB color format")
            if ENAVLE_CV2:
                image = image.transpose(1, 2, 0)
                image = cv2.cvtColor(image.astype(
                    np.uint8), cv2.COLOR_GRAY2RGB)
                image = image.transpose(2, 0, 1)
                image = image.astype(np.float32)

            else:
                new_image = np.zeros((3, 224, 224)).astype(np.float32)
                plane = image[0]
                new_image[0] = plane
                new_image[1] = plane
                new_image[2] = plane
                image = new_image
        return image, label
