"""i3d implemented in Gluon. WIP

For article "Quo Vadis..CVPR2017 deepmind
http://openaccess.thecvf.com/content_cvpr_2017/papers/Carreira_Quo_Vadis_Action_CVPR_2017_paper.pdf
https://github.com/deepmind/kinetics-i3d/blob/master/i3d.py

"""

import os

import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from mxnet.gluon.contrib.nn import HybridConcurrent


# helpers
def _make_unit3d(output_channels,
                 kernel_shape=(1, 1, 1),
                 strides=(1, 1, 1),
                 activation_fn='relu',
                 use_batch_norm=True,
                 use_bias=False,
                 layout='NDHWC',
                 **kwargs):
    # in tf default shape is NDHWC.
    # in gluon is NCDHW; num frames is depth; HXw is the image shape; C is the input channels
    # which is 3 for RGB and 2 for optical flow

    out = nn.HybridSequential(prefix='')
    out.add(nn.Conv3D(channels=output_channels,
                      kernel_size=kernel_shape,
                      strides=strides,
                      use_bias=use_bias,
                      layout=layout))
    if use_batch_norm:
        out.add(nn.BatchNorm(epsilon=0.001))
    if activation_fn is not None:
        out.add(nn.Activation(activation_fn))
    return out


# def make_aux(classes):
#     out = nn.HybridSequential(prefix='')
# TODO: auxiliary classifiers.

# inflated inception modules components
def _make_branch(use_pool, *conv_settings):
    out = nn.HybridSequential(prefix='')
    if use_pool == 'max':
        out.add(nn.MaxPool3D(pool_size=(1, 3, 3),
                             strides=(1, 2, 2)))
    setting_names = ['output_channels', 'kernel_shape', 'strides', 'padding']
    for setting in conv_settings:
        kwargs = {}
        for i, value in enumerate(setting):
            if value is not None:
                kwargs[setting_names[i]] = value
        out.add(_make_unit3d(**kwargs))
    return out

def _make_mixed_3b(pool_features, prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        # branch0
        out.add(_make_branch(None,
                             (64, 1, None, None)))
        # branch1
        out.add(_make_branch(None,
                             (96, 1, None, None),
                             (128, 3, None, None)))
        # branch2
        out.add(_make_branch(None,
                             (16, 1, None, None),
                             (32, 3, None, None)))

        # branch3
        # TODO: same pad here
        # out.add(_make_branch('max', (pool_features, 3, 1, None)),
        #         (32, 1, None, None))
        return out

class InceptionI3d(HybridBlock):

    def __init__(self, classes=4, dropout_keep_prob=0.5,
                 **kwargs):
        """400 classes in the Kinetics dataset."""
        super(InceptionI3d, self).__init__(**kwargs)
        self._num_classes = classes
        self.dropout_keep_prob = dropout_keep_prob

        # this is only the sequence ending in logits endpoint in i3d; the main classifier
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')

            #First layer Conv 7X7X7 stride 2
            # rgb input shape is 1X79X224X224X3; works with kernel = 3
            self.features.add(_make_unit3d(output_channels=64,
                                           kernel_shape=7,
                                           strides=2))
            # Todo: check shapes

            self.features.add(nn.MaxPool3D(pool_size=(1, 3, 3),
                                           strides=(1, 2, 2)))
            self.features.add(_make_unit3d(output_channels=64,
                                           kernel_shape=(3, 3, 3)
                                           ))
            self.features.add(_make_unit3d(output_channels=192,
                                           kernel_shape=(1, 1, 1)
                                           ))
            self.features.add(nn.MaxPool3D(pool_size=(1, 3, 3),
                                           strides=(1, 2, 2)))

            # TODO: check on padding

            # TODO: Inception modules (WIP)

            # self.features.add(_make_mixed_3b(32, 'mixed_3b'))

            # tail
            # self.features.add(nn.AvgPool3D(pool_size=1,
            #                                strides=(1, 1, 1, 1, 1)))
            # self.features.add(nn.AvgPool3D(pool_size=(1, 1, 1),
            #                                strides=(1, 1, 1, 1, 1))) # this has 5 dims on tf and valid padding
            self.features.add(nn.Dropout(self.dropout_keep_prob))
            self.features.add(_make_unit3d(output_channels=self._num_classes,
                                           kernel_shape=(1, 1, 1),
                                           activation_fn=None,
                                           use_batch_norm=False,
                                           use_bias=True))

            # logits/main classifier outputs endpoint
            self.output = nn.HybridSequential(prefix='')
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(self._num_classes))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

# Constructor
def i3d(pretrained=False, ctx=mx.cpu(), root='assets/models', **kwargs):
    net = InceptionI3d(**kwargs)
    if pretrained:
        from .assets import get_model_file
        net.load_parameters(get_model_file('i3d', root=root), ctx=ctx)
    return net
