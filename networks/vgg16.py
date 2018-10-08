import chainer
import chainer.links as L
import chainer.functions as F


class VGG16(chainer.Chain):

    def __init__(self, num_classes, **kwargs):
        super(VGG16, self).__init__()
        with self.init_scope():
            self.base = L.VGG16Layers()
            self.fc_6 = L.Linear(512 * 7 * 7, 2048)
            self.fc_7 = L.Linear(2048, 1024)
            self.fc_8 = L.Linear(1024, num_classes)

    def __call__(self, x):
        h = self.base(x, layers=["pool5"])["pool5"]
        h = F.dropout(F.relu(self.fc_6(h)))
        h = F.dropout(F.relu(self.fc_7(h)))
        h = self.fc_8(h)
        return h

    def disable_target_layers(self):
        disables = ['conv1_1', 'conv1_2',
                    'conv2_1', 'conv2_2',
                    #'conv3_1', 'conv3_2', 'conv3_3',
                    #'conv4_1', 'conv4_2', 'conv4_3',
                    ]
        for layer in disables:
            self.base[layer].disable_update()
