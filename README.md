# chainer-food101

Classify category of food with Chainer based on MobileNetV2

# What's NEW

- update README.md
- some bugfix
- add new model ResNet50
- add pretrained model based on ResNet50 and MobileNetV2

# How to Train
## Prepare Dataset

Download Food-101 Data Set: [Food-101 -- Mining Discriminative Components with Random Forests](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)

```
$ git clone git@github.com:terasakisatoshi/chainer-food-101.git
$ cd chainer-food101
$ wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
$ tar xfvz http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
```

Then the structure of directory under this repository should be...

```console
$ tree -d
.
├── food-101
│   ├── images
│   │   ├── apple_pie
│   │   ├── baby_back_ribs
│   │   ├── baklava
│   │   ├── beef_carpaccio
│   │   ├── beef_tartare
│   │   ├── beet_salad
│   │   ├── beignets
│   │   ├── bibimbap
│   │   ├── bread_pudding
│   │   ├── breakfast_burrito
│   │   ├── bruschetta
│   │   ├── caesar_salad
│   │   ├── cannoli
│   │   ├── caprese_salad
│   │   ├── carrot_cake
│   │   ├── ceviche
│   │   ├── cheesecake
│   │   ├── cheese_plate
│   │   ├── chicken_curry
│   │   ├── chicken_quesadilla
│   │   ├── chicken_wings
│   │   ├── chocolate_cake
│   │   ├── chocolate_mousse
│   │   ├── churros
│   │   ├── clam_chowder
│   │   ├── club_sandwich
│   │   ├── crab_cakes
│   │   ├── creme_brulee
│   │   ├── croque_madame
│   │   ├── cup_cakes
│   │   ├── deviled_eggs
│   │   ├── donuts
│   │   ├── dumplings
│   │   ├── edamame
│   │   ├── eggs_benedict
│   │   ├── escargots
│   │   ├── falafel
│   │   ├── filet_mignon
│   │   ├── fish_and_chips
│   │   ├── foie_gras
│   │   ├── french_fries
│   │   ├── french_onion_soup
│   │   ├── french_toast
│   │   ├── fried_calamari
│   │   ├── fried_rice
│   │   ├── frozen_yogurt
│   │   ├── garlic_bread
│   │   ├── gnocchi
│   │   ├── greek_salad
│   │   ├── grilled_cheese_sandwich
│   │   ├── grilled_salmon
│   │   ├── guacamole
│   │   ├── gyoza
│   │   ├── hamburger
│   │   ├── hot_and_sour_soup
│   │   ├── hot_dog
│   │   ├── huevos_rancheros
│   │   ├── hummus
│   │   ├── ice_cream
│   │   ├── lasagna
│   │   ├── lobster_bisque
│   │   ├── lobster_roll_sandwich
│   │   ├── macaroni_and_cheese
│   │   ├── macarons
│   │   ├── miso_soup
│   │   ├── mussels
│   │   ├── nachos
│   │   ├── omelette
│   │   ├── onion_rings
│   │   ├── oysters
│   │   ├── pad_thai
│   │   ├── paella
│   │   ├── pancakes
│   │   ├── panna_cotta
│   │   ├── peking_duck
│   │   ├── pho
│   │   ├── pizza
│   │   ├── pork_chop
│   │   ├── poutine
│   │   ├── prime_rib
│   │   ├── pulled_pork_sandwich
│   │   ├── ramen
│   │   ├── ravioli
│   │   ├── red_velvet_cake
│   │   ├── risotto
│   │   ├── samosa
│   │   ├── sashimi
│   │   ├── scallops
│   │   ├── seaweed_salad
│   │   ├── shrimp_and_grits
│   │   ├── spaghetti_bolognese
│   │   ├── spaghetti_carbonara
│   │   ├── spring_rolls
│   │   ├── steak
│   │   ├── strawberry_shortcake
│   │   ├── sushi
│   │   ├── tacos
│   │   ├── takoyaki
│   │   ├── tiramisu
│   │   ├── tuna_tartare
│   │   └── waffles
│   └── meta
└── pretrained
```

## Prepare Python Modules
- Python(3.6.5 Miniconda)
- NumPy(1.14.5) (import numpy as np)
- Matplotlib(2.2.3) (from matplotlib import pyplot as plt)
- Chainer(4.4.0) (import chainer)
- ChainerCV(0.10.0) (import chainercv)
- OpenCV(3.4.2) (import cv2)

## Getting Started

```console
python train.py --device 0 --epoch 100 --destination trained
```

It takes 24 hours to train on my machine:

```console
$ lsb_release -a
No LSB modules are available.
Distributor ID:	Ubuntu
Description:	Ubuntu 16.04.5 LTS
Release:	16.04
Codename:	xenial
$ julia -e "versioninfo()"
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: Intel(R) Xeon(R) CPU E5-2683 v3 @ 2.00GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-6.0.0 (ORCJIT, haswell)
$ lspci | grep -i nvidia
01:00.0 VGA compatible controller: NVIDIA Corporation Device 1b06 (rev a1)
01:00.1 Audio device: NVIDIA Corporation Device 10ef (rev a1)
06:00.0 VGA compatible controller: NVIDIA Corporation Device 1b06 (rev a1)
06:00.1 Audio device: NVIDIA Corporation Device 10ef (rev a1)
```

## Hard To Train ?

Do not worry, we've prepared pretrained model. See next Chapter.

# Evaluate Your Model

After training, you can check accuracy top_1 and top_5.
Run this script.

```console
$ python predict.py trained/model_epoch_100.npz --device 0
```

If you want to compare pretrained model, run this script.

```console
python predict.py -h
usage: predict.py [-h] [--sample SAMPLE] [--device DEVICE] [--dataset DATASET]
                  model_path

positional arguments:
  model_path         path/to/snapshot e.g. pretrained/model_epoch_100.npz

optional arguments:
  -h, --help         show this help message and exit
  --sample SAMPLE    select num of --sample from test dataset to evaluate
                     accuracy
  --device DEVICE    specify GPU_ID. If negative, use CPU
  --dataset DATASET
```

for example

```console
$ python predict.py ./mv2 --dataset path/to/food-101 --device 0
```

This should be get:

```
#./mv2
top1 accuracy 0.6074059405940594
top5 accuracy 0.8492673267326732
```

```
./resnet50
top1 accuracy 0.6994851485148514
top5 accuracy 0.8940594059405941
```

# Using Web Camera

Wanna try demo ? You can use video.py:

```
python video.py ./mv2 path/to/food-101
```

Enjoy!
