import chainer
import chainer.links as L
import chainer.functions as F


class VGG16(chainer.Chain):

    def __init__(self, num_classes, **kwargs):
        super(VGG16, self).__init__()
        with self.init_scope():
            self.base = L.VGG16Layers()
            self.fc_6 = L.Linear(None, 4096)
            self.fc_7 = L.Linear(None, 2048)
            self.fc_last = L.Linear(None, num_classes)

    def __call__(self, x):
        h = self.base(x, layers=['pool5'])['pool5']
        h = F.dropout(F.relu(self.fc_6(h)))
        h = F.dropout(F.relu(self.fc_7(h)))
        h = self.fc_last(h)
        return h
