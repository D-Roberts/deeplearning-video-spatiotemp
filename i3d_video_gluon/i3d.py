"""Barebones i3d implemented in Gluon.

For article "Quo Vadis..CVPR2017 deepmind
http://openaccess.thecvf.com/content_cvpr_2017/papers/Carreira_Quo_Vadis_Action_CVPR_2017_paper.pdf
https://github.com/deepmind/kinetics-i3d/blob/master/i3d.py

Main classifier, no auxiliary tasks endpoints.

"""

from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from mxnet.gluon.contrib.nn import HybridConcurrent

# helpers
def _make_unit3d(**kwargs):
    # in tf default shape is NDHWC.
    # in gluon is NCDHW; num frames is depth; HXW is the image shape; C is the input channels
    # which is 3 for RGB and 2 for optical flow
    # 1x79x224x224x3 input shape in tensorflow
    # kernel should be applied on DHW
    # input shape should be 1x3X79x224x224 in gluon

    out = nn.HybridSequential(prefix='')
    out.add(nn.Conv3D(use_bias=False,
                      **kwargs))
    out.add(nn.BatchNorm(epsilon=0.001))
    out.add(nn.Activation('relu'))
    return out

# inflated inception modules components
def _make_branch(use_pool=None, *conv_settings):
    out = nn.HybridSequential(prefix='')
    if use_pool == 'max':
        # 1 is for the depth dimension for video inflation
        out.add(nn.MaxPool3D(pool_size=(3, 3, 3),
                             strides=(1, 1, 1),
                             padding=(1, 1, 1)))
    setting_names = ['channels', 'kernel_size', 'strides', 'padding']
    for setting in conv_settings:
        kwargs = {}
        for i, value in enumerate(setting):
            if value is not None:
                kwargs[setting_names[i]] = value
        out.add(_make_unit3d(**kwargs))

    return out

def _make_mixed_3b(prefix):

    out = HybridConcurrent(axis=1, prefix=prefix)

    with out.name_scope():
        # branch0
        out.add(_make_branch(None,
                             (64, 1, None, None)))
        #(1, 64, 35, 107, 107)

        # branch1
        out.add(_make_branch(None,
                             (96, 1, None, None),
                             (128, 3, None, 1)))
        # branch2
        out.add(_make_branch(None,
                             (16, 1, None, None),
                             (32, 3, None, 1)))
        # branch3
        out.add(_make_branch('max',
                             (32, 1, None, None)))

        return out

def _make_mixed_3c(prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    # axis 1 still correct
    with out.name_scope():
        # branch0
        out.add(_make_branch(None,
                             (128, 1, None, None)))
        # branch1
        out.add(_make_branch(None,
                             (128, 1, None, None),
                             (192, 3, None, 1)))

        # branch 2
        out.add(_make_branch(None,
                             (32, 1, None, None),
                             (96, 3, None, 1)))

        # branch 3
        out.add(_make_branch('max',
                             (64, 1, None, None)
                             ))
    return out

def _make_mixed_4b(prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        # branch 0
        out.add(_make_branch(None,
                             (192, 1, None, None)))
        # branch 1
        out.add(_make_branch(None,
                             (96, 1, None, None),
                             (208, 3, None, 1)))
        # branch 2
        out.add(_make_branch(None,
                             (16, 1, None, None),
                             (48, 3, None, 1)))
        # branch 3
        out.add(_make_branch('max',
                             (64, 1, None, None),
                             ))
        return out

def _make_mixed_4c(prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        # branch 0
        out.add(_make_branch(None,
                             (160, 1, None, None)))
        # branch 1
        out.add(_make_branch(None,
                             (112, 1, None, None),
                             (224, 3, None, 1)))
        # branch 2
        out.add(_make_branch(None,
                             (24, 1, None, None),
                             (64, 3, None, 1)))
        # branch 3
        out.add(_make_branch('max',
                             (64, 1, None, None),
                             ))
        return out

def _make_mixed_4d(prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        # branch 0
        out.add(_make_branch(None,
                             (128, 1, None, None)))
        # branch 1
        out.add(_make_branch(None,
                             (128, 1, None, None),
                             (256, 3, None, 1)))
        # branch 2
        out.add(_make_branch(None,
                             (24, 1, None, None),
                             (64, 3, None, 1)))
        # branch 3
        out.add(_make_branch('max',
                             (64, 1, None, None),
                             ))
        return out

def _make_mixed_4e(prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        # branch 0
        out.add(_make_branch(None,
                             (112, 1, None, None)))
        # branch 1
        out.add(_make_branch(None,
                             (144, 1, None, None),
                             (288, 3, None, 1)))
        # branch 2
        out.add(_make_branch(None,
                             (32, 1, None, None),
                             (64, 3, None, 1)))
        # branch 3
        out.add(_make_branch('max',
                             (64, 1, None, None),
                             ))
        return out

def _make_mixed_4f(prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        # branch 0
        out.add(_make_branch(None,
                             (256, 1, None, None)))
        # branch 1
        out.add(_make_branch(None,
                             (160, 1, None, None),
                             (320, 3, None, 1)))
        # branch 2
        out.add(_make_branch(None,
                             (32, 1, None, None),
                             (128, 3, None, 1)))
        # branch 3
        out.add(_make_branch('max',
                             (128, 1, None, None),
                             ))
        return out

def _make_mixed_5b(prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        # branch 0
        out.add(_make_branch(None,
                             (256, 1, None, None)))
        # branch 1
        out.add(_make_branch(None,
                             (160, 1, None, None),
                             (320, 3, None, 1)))
        # branch 2
        out.add(_make_branch(None,
                             (32, 1, None, None),
                             (128, 3, None, 1)))
        # branch 3
        out.add(_make_branch('max',
                             (128, 1, None, None),
                             ))
        return out

def _make_mixed_5c(prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        # branch 0
        out.add(_make_branch(None,
                             (384, 1, None, None)))
        # branch 1
        out.add(_make_branch(None,
                             (192, 1, None, None),
                             (384, 3, None, 1)))
        # branch 2
        out.add(_make_branch(None,
                             (48, 1, None, None),
                             (128, 3, None, 1)))
        # branch 3
        out.add(_make_branch('max',
                             (128, 1, None, None),
                             ))
        return out

class InceptionI3d(HybridBlock):

    def __init__(self, classes=4, dropout_keep_prob=0.5,
                 **kwargs):
        """400 classes in the Kinetics dataset."""
        super(InceptionI3d, self).__init__(**kwargs)
        self._num_classes = classes
        self.dropout_keep_prob = dropout_keep_prob


        # this is the main classifier
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')

            # the input shape is `batch_size` x `num_frames` x 224 x 224 x `num_channels` in tf code
            # but gluon is NCDHW
            # input shape is 1, 3, 79, 224, 224

            self.features.add(_make_unit3d(channels=64,
                                           kernel_size=(7, 7, 7),
                                           strides=(2, 2, 2)))
            # shape is (1, 64, 37, 109, 109)

            self.features.add(nn.MaxPool3D(pool_size=(1, 3, 3),
                                           strides=(1, 2, 2),
                                           padding=(0, 55, 55))) # here should be 'same' padding; hard code for now.
            # shape is (1, 64, 37, 109, 109)

            self.features.add(_make_unit3d(channels=64,
                                           kernel_size=(1, 1, 1)
                                           ))
            # shape (1, 64, 37, 109, 109)

            self.features.add(_make_unit3d(channels=192,
                                           kernel_size=(3, 3, 3)
                                           ))
            # shape (1, 192, 35, 107, 107)

            self.features.add(nn.MaxPool3D(pool_size=(1, 3, 3),
                                           strides=(1, 2, 2),
                                           padding=(0, 54, 54))) # padding same
            # shape (1, 192, 35, 107, 107)

            self.features.add(_make_mixed_3b('mixed_3b'))

            self.features.add(_make_mixed_3c('mixed_3c'))
            #(1, 480, 35, 107, 107)

            self.features.add(nn.MaxPool3D(pool_size=(3, 3, 3),
                                           strides=(2, 2, 2),
                                           padding=(18, 54, 54))) # padding is same here

            self.features.add(_make_mixed_4b('mixed_4b'))
            #
            self.features.add(_make_mixed_4c('mixed_4c'))

            self.features.add(_make_mixed_4d('mixed_4d'))

            self.features.add(_make_mixed_4e('mixed_4e'))

            self.features.add(_make_mixed_4f('mixed_4f'))
             # (1, 384, 35, 107, 107)

            self.features.add(nn.MaxPool3D(pool_size=(2, 2, 2),
                                           strides=(2, 2, 2),
                                           padding=(18, 54, 54)))

            self.features.add(_make_mixed_5b('mixed_5b'))

            self.features.add(_make_mixed_5c('mixed_5c'))

            self.features.add(nn.AvgPool3D(pool_size=(2, 7, 7)))

            self.features.add(nn.Dropout(self.dropout_keep_prob))

            self.features.add(_make_unit3d(channels=self._num_classes,
                                           kernel_size=(1, 1, 1)))

            # logits/main classifier outputs endpoint
            self.output = nn.HybridSequential(prefix='')
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(self._num_classes))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

# Constructor
def i3d(**kwargs):
    net = InceptionI3d(**kwargs)
    return net