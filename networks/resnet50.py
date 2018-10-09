import chainer
import chainer.links as L
import chainer.functions as F


class ResNet50(chainer.Chain):

    def __init__(self, num_classes, **kwargs):
        super(ResNet50, self).__init__()
        with self.init_scope():
            self.base = L.ResNet50Layers()
            self.fc_1 = L.Linear(None, 1024)
            self.fc_2 = L.Linear(1024, num_classes)

    def __call__(self, x):
        h = self.base(x, layers=["pool5"])["pool5"]
        h = F.dropout(F.relu(self.fc_1(h)))
        h = self.fc_2(h)
        return h

    def disable_target_layers(self):
        disables = ['conv1',
                    'res2',
                    'res3',
                    'res4',
                    #'res5',
                    ]

        for layer in disables:
            self.base[layer].disable_update()


def main():
    resnet = ResNet50(num_classes=101)
    print(resnet.base.available_layers)

if __name__ == '__main__':
    main()
