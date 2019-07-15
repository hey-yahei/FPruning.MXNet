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

__all__ = ['Pruner', 'PrunerManager']
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


class Pruner(object):
    def __init__(self, pruned_conv, mask_output, share_mask=None):
        """
        Filter-level pruner for convolution layers.
        :param pruned_conv: mxnet.gluon.nn.Conv2D
            Convolution block to prune.
        :param mask_output: mxnet.gluon.nn.Conv2D, or mxnet.gluon.nn.BatchNorm
            Convolution or BatchNorm block whose outputs are applied mask to.
        :param share_mask: mxnet.gluon.nn.Conv2D
            If not None, share mask with specified Convolution block.
        """
        """ Store basic attributes """
        self.pruned_conv = pruned_conv
        self.mask_output = mask_output
        self.share_mask = share_mask

        """ Initialize a mask if not share """
        if share_mask is None:
            weight = pruned_conv.weight
            self.mask = nd.ones(shape=(1, weight.shape[0], 1, 1), ctx=weight.list_ctx()[0])
        else:
            self.mask = None

        """ Apply mask to outputs of specified block"""
        def _forward(self_, *args, **kwargs):
            mask = self.share_mask.mask if self.share_mask is not None else self.mask
            out = self_.origin_forward(*args, **kwargs)
            return _ChannelMask(mask)(out)
        mask_output.origin_forward = mask_output.hybrid_forward
        mask_output.hybrid_forward = types.MethodType(_forward, mask_output)

    def analyse(self, out_size):
        """
        Analyse the results for pruning.
        :param out_size: (out_height, out_width)
            out_height is the height of output feature map of self.mask_output, while out_width the width.
        :return: ((pruned_channels, total_channels), (pruned_params, total_params), (pruned_mac, total_mac))
            pruned_channels: int, the number of channels in pruned model
            total_channels: int, the number of channels in origin model
            pruned_params: int, the number of parameters in pruned model
            total_params: int, the number of parameters in origin model
            pruned_mac: int, the number of MAC(Multiply-ACcumulator) in pruned model
            total_mac: int, the number of MAC(Multiply-ACcumulator) in origin model
        """
        # Calculate the number of parameters
        oc, ic, kh, kw = self.pruned_conv.weight.shape
        pc = oc - self.mask.sum().asscalar()
        total_params = oc * ic * kh * kw
        pruned_params = pc * ic * kh * kw
        # Calculate the MAC
        oh, ow = out_size
        total_mac = oh * ow * oc * kw * kh * ic
        pruned_mac = oh * ow * pc * kw * kh * ic

        return (pc, oc), (pruned_params, total_params), (pruned_mac, total_mac)

    def default_prune(self):
        """ The default pruning API """
        raise NotImplementedError()


class PrunerManager(object):
    def __init__(self, net):
        """
        A manager for Pruners
        :param net: mxnet.gluon.nn.HybridBlock
            the net to be pruned.
        """
        # Collection for Pruners
        self.pruner_list = []
        # Collection for sizes of output feature map
        self.out_size = {}
        # Net
        self._net = net

    def build(self, in_shape):
        """
        Build manager after configuration.
        :param in_shape: (out_channels, in_channels, in_height, in_width)
            the shape of input for net
        """
        # Map Conv2D to corresponding Pruner for share_mask in Pruner
        mapper = {pruner.pruned_conv: pruner for pruner in self.pruner_list}
        for pruner in self.pruner_list:
            conv = pruner.share_mask
            if conv is not None:
                pruner.share_mask = mapper[conv]
        # Infer sizes of output feature map
        self._get_outsize(in_shape)

    def add(self, pruner):
        """
        Add a pruner to manager.
        :param pruner: Pruner
        """
        self.pruner_list.append(pruner)

    def compose(self, *pruners):
        """
        Add several pruners to manager
        :param pruners: Pruners
        """
        self.pruner_list.extend(pruners)

    def apply(self, func):
        """
        Apply func to every pruners, like mxnet.gluon.nn.HybridBlock.apply()
        :param func: func(pruner) -> None
        """
        for pruner in self.pruner_list:
            func(pruner)

    def prune(self, *args, **kwargs):
        """ Apply prune via default_prune APIs """
        for pruner in self.pruner_list:
            pruner.default_prune(*args, **kwargs)

    def analyse(self):
        """
        Analyse the results for pruning.
        :return: (param_sparsity, mac_sparsity)
            param_sparsity: pruned% for parameters
            mac_sparsity: pruned% for MAC
        """
        assert self.out_size is not None, "Please run get_outsize() to collect output shape of convolutions."

        pruned_params, total_params, pruned_mac, total_mac = 0, 0, 0, 0
        for pruner in self.pruner_list:
            _, (pp, tp), (pm, tm) = pruner.analyse(self.out_size[pruner])
            pruned_params += pp
            total_params += tp
            pruned_mac += pm
            total_mac += tm

        return pruned_params / total_params, pruned_mac / total_mac

    def _get_outsize(self, in_shape):
        """ Collect the output shape of feature maps """
        hooks = []
        for pruner in self.pruner_list:
            def _generate_hook(pruner):
                def _hook(m, x, y):
                    shape = y.shape
                    self.out_size[pruner] = (shape[2], shape[3])
                return _hook
            h = pruner.mask_output.register_forward_hook(_generate_hook(pruner))
            hooks.append(h)

        ctx = self.pruner_list[0].pruned_conv.weight.list_ctx()[0]
        in_ = nd.zeros(shape=in_shape, ctx=ctx)
        _ = self._net(in_)

        for h in hooks:
            h.detach()
