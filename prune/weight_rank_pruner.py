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

from .pruner import Pruner

__all__ = ['WeightRankPruner', 'L1RankPruner']
__author__ = 'YaHei'

WeightRankPruner = Pruner


class L1RankPruner(WeightRankPruner):
    """ Reference: https://arxiv.org/abs/1608.08710 """
    def __init__(self, pruned_conv, mask_output, share_mask=None):
        """ L1-rank pruner, refer to Pruner """
        super(L1RankPruner, self).__init__(pruned_conv, mask_output, share_mask)
        self.default_prune = self.prune_by_std

    def prune_by_std(self, s=0.25):
        """
        Prune with the sensitivity as threshold for L1-rank.
        $sensitivity = std(weight) * s$
        :param s: float
            the factor in sensitivity
        """
        ctx = self.pruned_conv.weight.list_ctx()[0]
        weight = self.pruned_conv.weight.data()
        th = np.std(weight.asnumpy()) * s
        th = nd.array([th], ctx=ctx)
        abs_mean = weight.abs().mean(axis=(1, 2, 3))
        self.mask = (abs_mean >= th).reshape(1, -1, 1, 1)