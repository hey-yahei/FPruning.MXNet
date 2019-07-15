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

from mxnet import autograd, nd

from .pruner import Pruner

__all__ = ['GradientRankPruner', 'GradientTaylorRankPruner', 'GradientWeightRankPruner']
__author__ = 'YaHei'


class _CollectOutputGradient(autograd.Function):
    def __init__(self, out_y, out_dy):
        super(_CollectOutputGradient, self).__init__()
        self.out_y = out_y
        self.out_dy = out_dy

    def forward(self, y):
        self.out_y.clear()
        self.out_y.append(y)

    def backward(self, dy):
        self.out_dy.clear()
        self.out_dy.append(dy)


class GradientRankPruner(Pruner):
    def __init__(self, pruned_conv, mask_output):
        super(GradientRankPruner, self).__init__(pruned_conv, mask_output)


class GradientTaylorRankPruner(GradientRankPruner):
    """ Reference: https://arxiv.org/abs/1611.06440 """
    def __init__(self, pruned_conv, mask_output):
        super(GradientTaylorRankPruner, self).__init__(pruned_conv, mask_output)
        self.default_prune = self.prune_by_percent

        self.taylors = []

        self.y = []
        self.dy = []
        def _forward(self_, *args, **kwargs):
            out = self_.origin_forward(*args, **kwargs)
            return _CollectOutputGradient(self.y, self.dy)(out)
        pruned_conv.origin_forward = pruned_conv.hybrid_forward
        pruned_conv.hybrid_forward = types.MethodType(_forward, pruned_conv)

    def update_state(self):
        y = self.y[0].asnumpy()
        dy = self.dy[0].asnumpy()
        taylor = (y * dy).mean(axis=(2, 3))
        self.taylors.append(taylor)

    def _compute_mean_taylor_and_clear(self):
        taylors = np.concatenate(self.taylors, axis=0)
        mean = taylors.mean(axis=0)
        self.taylors.clear()
        return abs(mean)

    def prune_by_percent(self, p):
        ctx = self.pruned_conv.weight.list_ctx()[0]
        taylors = self._compute_mean_taylor_and_clear()
        th_idx = np.argsort(taylors)[int(p * self._channels)]
        mask = (taylors >= th_idx).reshape(1, -1, 1, 1)
        self.mask = nd.array(mask, ctx=ctx)


class GradientWeightRankPruner(GradientRankPruner):
    """ Reference: https://github.com/NervanaSystems/distiller/blob/master/distiller/pruning/ranked_structures_pruner.py#L521"""
    def __init__(self, pruned_conv, mask_output):
        super(GradientWeightRankPruner, self).__init__(pruned_conv, mask_output)
        self.default_prune = self.prune_by_percent

    def _compute_criterion(self):
        weight = self.pruned_conv.weight.data()
        grad = self.pruned_conv.weight.grad()
        criterion = (weight * grad).mean(axis=(1, 2, 3))
        return abs(criterion)

    def prune_by_percent(self, p):
        criterion = self._compute_criterion()
        th_idx = nd.argsort(criterion)[int(p * self._channels)]
        self.mask = (criterion >= th_idx).reshape(1, -1, 1, 1)

