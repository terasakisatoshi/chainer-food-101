import json
import logging
import os
import random

from PIL import Image
import chainer
from chainer.datasets import LabeledImageDataset
from chainer.datasets import TransformDataset
from chainercv import transforms
from chainer.links.model.vision import vgg
from chainer.links.model.vision import resnet
try:
    import cv2
    ENAVLE_CV2 = True
except ImportError:
    ENAVLE_CV2 = False

import numpy as np

logger = logging.getLogger(__name__)


BLACKLIST = ["lasagna/3787908",
             "steak/1340977",
             "bread_pudding/1375816"]


DEG_RANGE = np.linspace(-20, 20, 100)


def rotate_image(image):
    pilimg = Image.fromarray(image.transpose(1, 2, 0).astype(np.uint8))
    pilimg = pilimg.rotate(np.random.choice(DEG_RANGE))
    return np.asarray(pilimg).transpose(2, 0, 1)


def preprocess(image, model_name):
    if model_name == "mv2":
        image /= 128.
        image = transforms.resize(image, (224, 224))
    elif model_name == "vgg16":
        image = vgg.prepare(image, size=(224, 224))
    elif model_name == "resnet50":
        image = resnet.prepare(image, size=(224, 224))
    else:
        raise Exception("illegal model")
    return image


def get_pairs(dataset_dir, train=True):
    meta_dir = os.path.join(dataset_dir, "meta")
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
            pairs.append((os.path.join(dataset_dir, "images", p + ".jpg"),
                          label))
    return pairs


class FoodDataset(chainer.dataset.DatasetMixin):

    def __init__(self, dataset_dir, model_name, train=True):
        pairs = get_pairs(dataset_dir, train=train)
        self.base = LabeledImageDataset(pairs)
        self.train = train
        self.pairs = pairs
        self.model_name = model_name

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        imgpath = self.base._pairs[i][0]
        image = image.copy().astype(np.float32)
        if self.train:
            image = rotate_image(image)
            image = transforms.resize(image,
                                      (random.choice(range(368, 512)), random.choice(range(368, 512))))
            image = transforms.pca_lighting(image, 76.5)
            image = transforms.random_flip(image, x_random=True, y_random=True)
            if image.shape[1] >= 224 and image.shape[2] >= 224:
                image = transforms.random_crop(image, size=(224, 224))
        image = transforms.resize(image, (224, 224))
        image = preprocess(image, self.model_name)
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
