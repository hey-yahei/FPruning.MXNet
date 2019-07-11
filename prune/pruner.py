#-*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2019 hey-yahei
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import types
import numpy as np

from mxnet import nd, autograd
from mxnet.gluon import nn

from . import utils

__all__ = ['PrunerManager']
__author__ = 'YaHei'


class _ChannelMask(autograd.Function):
    """ Mask for channels not only for forward but also for backward """
    def __init__(self, mask):
        super(_ChannelMask, self).__init__()
        self.mask = mask

    def forward(self, x):
        return x * self.mask

    def backward(self, dy):
        return dy * self.mask


class PrunerManager(object):
    def __init__(self, net, exclude=[], share_mask={}):
        # Collection for masks of net
        masks = {}
        # Collection for  convolutions which have been dealed with
        visited_conv = []
        # Collect convoluion-bn pairs
        conv_bn = utils.get_conv_bn_pairs(net)

        """ Firstly, add mask to all batchnorm """
        def _add_mask_to_bn(m):
            nonlocal masks, share_mask, visited_conv, conv_bn
            if isinstance(m, nn.BatchNorm):
                # Get corresponding convolution
                conv = conv_bn.get_conv(m)
                # If exclude such a convoluton
                if conv in exclude:
                    return
                # Create a init mask if not share
                if m not in share_mask:
                    channels = m.gamma.shape[0]
                    masks[m] = nd.ones(shape=(1, channels, 1, 1), ctx=m.gamma.list_ctx()[0])
                # Reset forward function
                def _forward(self_, *args, **kwargs):
                    nonlocal masks, share_mask
                    # Normal forward
                    out = self_.origin_forward(*args, **kwargs)
                    # Get mask and apply to outputs
                    share = share_mask.get(m)
                    mask = masks[m] if share is None else masks[share]
                    return _ChannelMask(mask)(out)
                m.origin_forward = m.hybrid_forward
                m.hybrid_forward = types.MethodType(_forward, m)
                # Add convolution to visited list
                visited_conv.append(conv)
        _ = net.apply(_add_mask_to_bn)

        """ Secondly, add mask to other convolutions """
        def _add_mask_to_other_conv(m):
            nonlocal masks, share_mask, visited_conv
            if all((isinstance(m, nn.Conv2D), m not in exclude, m not in visited_conv)):
                # Create a init mask if not share
                if m not in share_mask:
                    channels = m.weight.shape[0]
                    masks[m] = nd.ones(shape=(1, channels, 1, 1), ctx=m.weight.list_ctx()[0])

                # Reset forward function
                def _forward(self_, *args, **kwargs):
                    nonlocal masks, share_mask
                    # Normal forward
                    out = self_.origin_forward(*args, **kwargs)
                    # Get mask and apply to outputs
                    share = share_mask.get(m)
                    mask = masks[m] if share is None else masks[share]
                    return _ChannelMask()(out, mask)
                m.origin_forward = m.hybrid_forward
                m.hybrid_forward = types.MethodType(_forward, m)
        _ = net.apply(_add_mask_to_other_conv)
        # Store attributes
        self.masks = masks
        self.share_masks = share_mask
        self.conv_bn = conv_bn

    def prune_by_threshold(self, th):
        """ Prune filters with threshold """
        for m, mask in self.masks.items():
            conv = self.conv_bn.get_conv(m) if isinstance(m, nn.BatchNorm) else m
            weight = conv.weight.data().asnumpy()
            abs_mean = abs(weight).mean(axis=(1, 2, 3))
            mask = (abs_mean >= th).astype("float32")
            self.masks[m] = nd.array(mask, ctx=conv.weight.list_ctx()[0]).reshape(1, -1, 1, 1)

    def prune_by_percent(self, per):
        """ Prune filters with percent """
        for m, mask in self.masks.items():
            conv = self.conv_bn.get_conv(m) if isinstance(m, nn.BatchNorm) else m
            weight = conv.weight.data().asnumpy()
            abs_mean = abs(weight).mean(axis=(1, 2, 3))
            th_idx = np.argsort(abs_mean)[int(per * abs_mean.shape[0])]
            mask = (abs_mean >= abs_mean[th_idx]).astype("float32")
            self.masks[m] = nd.array(mask, ctx=conv.weight.list_ctx()[0]).reshape(1, -1, 1, 1)
