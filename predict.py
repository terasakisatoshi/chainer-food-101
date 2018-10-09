import argparse
from glob import glob
import os
import json
import random

import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np

from networks.mobilenetv2 import MobilenetV2
from networks.vgg16 import VGG16
from networks.resnet50 import ResNet50
from dataset import FoodDataset


def find_latest(model_dir):
    files = glob(os.path.join(model_dir, "model_epoch_*.npz"))
    numbers = []
    for f in files:
        base = os.path.basename(f)
        base = base[len("model_epoch_"):]
        base = base[:-len(".npz")]
        e = int(base)
        numbers.append(e)
    e = max(numbers)
    return os.path.join(model_dir, "model_epoch_{}.npz".format(e))


def prepare_setting(args):
    model_path = find_latest(args.model_path)
    print(model_path)
    jsonpath = os.path.join(os.path.dirname(model_path), "args.json")
    print(jsonpath)
    with open(jsonpath, 'r') as f:
        train_args = json.load(f)

    model_cand = {"mv2": MobilenetV2,
                  "vgg16": VGG16,
                  "resnet50": ResNet50}
    if train_args["model_name"] == "mv2":
        model = MobilenetV2(num_classes=101, depth_multiplier=1.0)
    else:
        model = model_cand[train_args["model_name"]](num_classes=101)

    model = L.Classifier(model)
    chainer.serializers.load_npz(model_path, model)

    test_dataset = FoodDataset(args.dataset,
                               model_name=train_args["model_name"],
                               train=False)

    if args.device >= 0:
        # use GPU
        chainer.backends.cuda.get_device_from_id(args.device).use()
        model.predictor.to_gpu()
        import cupy as xp
    else:
        # use CPU
        xp = np
    return model, xp, test_dataset


def predict(args):
    classes = np.genfromtxt(os.path.join(args.dataset, "meta", "labels.txt"),
                            str,
                            delimiter="\n")
    model, xp, test_dataset = prepare_setting(args)

    top_1_counter = 0
    top_5_counter = 0
    top_10_counter = 0
    indices = list(range(len(test_dataset)))
    num_iteration = len(indices) if args.sample < 0 else args.sample
    random.shuffle(indices)
    with chainer.function.no_backprop_mode(), chainer.using_config('train', False):
        for i in indices[:num_iteration]:
            img, label = test_dataset.get_example(i)
            h = model.predictor(xp.expand_dims(xp.array(img), axis=0))
            prediction = chainer.functions.softmax(h)
            if args.device >= 0:
                prediction = xp.asnumpy(prediction[0].data)
            else:
                prediction = prediction[0].data
            top_ten = np.argsort(-prediction)[:10]
            top_five = top_ten[:5]
            if top_five[0] == label:
                top_1_counter += 1
                top_5_counter += 1
                top_10_counter += 1
                msg = "Bingo!"
            elif label in top_five:
                top_5_counter += 1
                top_10_counter += 1
                msg = "matched top 5"
            elif label in top_ten:
                top_10_counter += 1
                msg = "matched top 10"
            else:
                msg = "Boo, actual {}".format(classes[label])
            print(classes[top_five], prediction[top_five], msg)
        print('top1 accuracy', top_1_counter / num_iteration)
        print('top5 accuracy', top_5_counter / num_iteration)
        print('top10 accuracy', top_10_counter / num_iteration)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path", type=str, help="path/to/snapshot e.g. pretrained/model_epoch_100.npz")
    parser.add_argument("--sample", type=int, default=-1,
                        help="select num of --sample from test dataset to evaluate accuracy")
    parser.add_argument("--device", type=int, default=0,
                        help="specify GPU_ID. If negative, use CPU")
    parser.add_argument("--dataset", type=str, default=".")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_argument()
    predict(args)
