import chainer
import chainer.functions as F
import chainer.links as L


def _make_divisible(v, divisor, min_value=None):
    """
    taken from https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/conv_blocks.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def expand_input_by_factor(n, divisible_by=8):
    return lambda num_inputs, **_: _make_divisible(num_inputs * n, divisible_by)


def multiplier(n, divisible_by, min_depth):
    return lambda num_inputs: _make_divisible(num_inputs * n, divisible_by, min_depth)


def relu6(x):
    return F.clipped_relu(x, z=6.0)


class Conv(chainer.Chain):

    def __init__(self, in_ch, out_ch, ksize, stride):
        super(Conv, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_ch,
                                        out_ch,
                                        ksize=ksize,
                                        stride=stride,
                                        pad=ksize // 2,
                                        nobias=True)
            self.bn = L.BatchNormalization(out_ch)

    def __call__(self, x):
        h = relu6(self.bn(self.conv(x)))
        return h


class ExpandedConv(chainer.Chain):

    def __init__(self, expand_input, in_ch, out_ch, stride):
        super(ExpandedConv, self).__init__()
        if expand_input == 1:
            expander = expand_input_by_factor(expand_input, 1)
        else:
            expander = expand_input_by_factor(expand_input, 8)
        self.in_ch, self.out_ch = in_ch, out_ch
        expanded_ch = expander(in_ch)
        self.stride = stride
        with self.init_scope():
            self.expand_conv = L.Convolution2D(in_ch,
                                               expanded_ch,
                                               ksize=1,
                                               nobias=True)
            self.expand_bn = L.BatchNormalization(expanded_ch)

            self.depthwise_conv = L.DepthwiseConvolution2D(expanded_ch,
                                                           channel_multiplier=1,
                                                           ksize=3,
                                                           stride=self.stride,
                                                           pad=1,
                                                           nobias=True)
            self.depthwise_bn = L.BatchNormalization(expanded_ch)

            self.project_conv = L.Convolution2D(expanded_ch,
                                                out_ch,
                                                ksize=1,
                                                nobias=True)
            self.project_bn = L.BatchNormalization(out_ch)

    def __call__(self, x):
        h = x
        h = relu6(self.expand_bn(self.expand_conv(h)))
        h = relu6(self.depthwise_bn(self.depthwise_conv(h)))
        h = self.project_bn(self.project_conv(h))
        if self.stride == 1 and self.in_ch == self.out_ch:
            h += x
        return h


class MobilenetV2(chainer.Chain):

    def __init__(self, num_classes, **kwargs):
        depth_multiplier = kwargs["depth_multiplier"]
        super(MobilenetV2, self).__init__()
        d = multiplier(depth_multiplier, divisible_by=8, min_depth=8)
        with self.init_scope():
            self.conv2d_begin = Conv(None, d(32), ksize=3, stride=2)

            self.expanded_conv1 = ExpandedConv(1, d(32), d(16), stride=1)

            self.expanded_conv2 = ExpandedConv(6, d(16), d(24), stride=2)
            self.expanded_conv3 = ExpandedConv(6, d(24), d(24), stride=1)

            self.expanded_conv4 = ExpandedConv(6, d(24), d(32), stride=2)
            self.expanded_conv5 = ExpandedConv(6, d(32), d(32), stride=1)
            self.expanded_conv6 = ExpandedConv(6, d(32), d(32), stride=1)

            self.expanded_conv7 = ExpandedConv(6, d(32), d(64), stride=2)
            self.expanded_conv8 = ExpandedConv(6, d(64), d(64), stride=1)
            self.expanded_conv9 = ExpandedConv(6, d(64), d(64), stride=1)
            self.expanded_conv10 = ExpandedConv(6, d(64), d(64), stride=1)

            self.expanded_conv11 = ExpandedConv(6, d(64), d(96), stride=1)
            self.expanded_conv12 = ExpandedConv(6, d(96), d(96), stride=1)
            self.expanded_conv13 = ExpandedConv(6, d(96), d(96), stride=1)

            self.expanded_conv14 = ExpandedConv(6, d(96), d(160), stride=2)
            self.expanded_conv15 = ExpandedConv(6, d(160), d(160), stride=1)
            self.expanded_conv16 = ExpandedConv(6, d(160), d(160), stride=1)

            self.expanded_conv17 = ExpandedConv(6, d(160), d(320), stride=1)
            self.conv2d_last = Conv(d(320), d(120), ksize=1, stride=1)
            self.fc = L.Linear(d(120), num_classes)

    def forward(self, x):
        h = self.conv2d_begin(x)
        h = self.expanded_conv1(h)
        h = self.expanded_conv2(h)
        h = self.expanded_conv3(h)
        h = self.expanded_conv4(h)
        h = self.expanded_conv5(h)
        h = self.expanded_conv6(h)
        h = self.expanded_conv7(h)
        h = self.expanded_conv8(h)
        h = self.expanded_conv9(h)
        h = self.expanded_conv10(h)
        h = self.expanded_conv11(h)
        h = self.expanded_conv12(h)
        h = self.expanded_conv13(h)
        h = self.expanded_conv14(h)
        h = self.expanded_conv15(h)
        h = self.expanded_conv16(h)
        h = self.expanded_conv17(h)
        h = self.conv2d_last(h)
        h = self.fc(F.average_pooling_2d(h, 7))
        return h
