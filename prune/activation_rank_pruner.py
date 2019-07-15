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

import numpy as np

from mxnet import autograd, nd

from .pruner import Pruner

__all__ = ['ActivationRankPruner', 'ActivationAPoZRankPruner', 'ActivationEntropyRankPruner']
__author__ = 'YaHei'


class ActivationRankPruner(Pruner):
    def __init__(self, pruned_conv, mask_output, act_blk):
        super(ActivationRankPruner, self).__init__(pruned_conv, mask_output)
        self.act_blk = act_blk


class ActivationAPoZRankPruner(ActivationRankPruner):
    """ Reference: https://arxiv.org/abs/1607.03250 """
    def __init__(self, pruned_conv, mask_output, act_blk):
        """ APoZ-rank pruner, refer to Pruner """
        super(ActivationAPoZRankPruner, self).__init__(pruned_conv, mask_output, act_blk)
        self.default_prune = self.prune_by_percent

        self.clear_state()
        def _hook(m, x, y):
            if not autograd.is_training():
                batch_mean = (y == 0).mean(axis=(2, 3)).asnumpy()
                for mean in batch_mean:
                    self.APoZs = 0.01 * mean + 0.99 * self.APoZs
        act_blk.register_forward_hook(_hook)

    def clear_state(self):
        """
        Clear APoZ list.
        Called at the begin of evaluation.
        """
        self.APoZs = np.zeros(shape=self._channels)

    def prune_by_percent(self, p):
        """
        Prune filters by ranked with p percent.
        Larger APoZ means less important.
        :param p: float < 1.0
            The percent of filters to prune.
        """
        ctx = self.pruned_conv.weight.list_ctx()[0]
        APoZs = np.concatenate(self.APoZs, axis=0)
        mean = APoZs.mean(aixs=0)
        th_idx = np.argsort(mean)[int((1-p) * self._channels)]
        mask = (mean < mean[th_idx]).reshape(1, -1, 1, 1)
        self.mask = nd.array(mask, ctx=ctx)


class ActivationEntropyRankPruner(ActivationRankPruner):
    """ Reference: http://arxiv.org/abs/1706.05791 """
    def __init__(self, pruned_conv, mask_output, act_blk):
        """ Entropy-rank pruner, refer to Pruner """
        super(ActivationEntropyRankPruner, self).__init__(pruned_conv, mask_output, act_blk)
        self.default_prune = self.prune_by_percent

        self.global_means = []
        def _hook(m, x, y):
            if not autograd.is_training():
                mean = y.mean(axis=(2, 3)).asnumpy()
                self.global_means.append(mean)
        act_blk.register_forward_hook(_hook)

    def _compute_entropy_and_clear(self, bins=100):
        global_mean_np = np.concatenate(self.global_means, axis=0)
        min_ = np.min(global_mean_np, axis=0, keepdims=True)
        max_ = np.max(global_mean_np, axis=0, keepdims=True)
        scales = bins / (max_ - min_)
        data2bin = ((global_mean_np - min_) * scales).astype("int32")

        entropys = []
        for bin in data2bin.swapaxes(0, 1):
            count = np.bincount(bin, minlength=bins)
            prob = count.astype("float32") / bin.size
            entropys.append(-sum(prob * np.log(prob)))

        self.global_means.clear()
        return np.array(entropys)

    def prune_by_percent(self, p, bins=100):
        """
        Prune filters by ranked with p percent.
        Larger entropy means more important.
        :param p: float < 1.0
            The percent of filters to prune.
        :param bins: int
            The number of bins to calculate probability distribution.
        """
        ctx = self.pruned_conv.weight.list_ctx()[0]
        entropys = self._compute_entropy_and_clear(bins)
        th_idx = np.argsort(entropys)[int(p * self._channels)]
        mask = (entropys >= th_idx).reshape(1, -1, 1, 1)
        self.mask = nd.array(mask, ctx=ctx)

