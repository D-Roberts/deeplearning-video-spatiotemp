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
-- INSERT --
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
-- INSERT --
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
-- INSERT --
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
-- INSERT --
dsd"""i3d implemented in Gluon. WIP

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

