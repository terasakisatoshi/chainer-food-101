import chainer
import chainer.links as L
import chainer.functions as F


class ResNet50(chainer.Chain):

    def __init__(self, num_classes, **kwargs):
        super(ResNet50, self).__init__()
        with self.init_scope():
            self.base = L.ResNet50Layers()
            self.fc_last = L.Linear(None, num_classes)

    def __call__(self, x):
        h = self.base(x, layers=["pool5"])["pool5"]
        h = self.fc_last(h)
        return h

    def disable_target_layers(self):
        disables = ['conv1',
                    'res2',
                    'res3',
                    #'res4',
                    #'res5',
                    ]

        for layer in disables:
            self.base[layer].disable_update()


def main():
    resnet = ResNet50(num_classes=101)
    print(resnet.base.available_layers)

if __name__ == '__main__':
    main()
