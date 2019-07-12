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

import gluoncv

__all__ = ['cifar10_resnet56_v1']
__author__ = 'YaHei'

def cifar10_resnet56_v1(net):
    assert isinstance(net, gluoncv.model_zoo.cifarresnet.CIFARResNetV1)

    convs = [net.features[0]] + \
            [sequential.body[i] for j in (2, 3, 4) for sequential in net.features[j] for i in (0, 3)] + \
            [net.features[j][0].downsample[0] for j in (3, 4)]
    bns = [net.features[1]] + \
          [sequential.body[i] for j in (2, 3, 4) for sequential in net.features[j] for i in (1, 4)] + \
          [net.features[j][0].downsample[1] for j in (3, 4)]
    shares = {
        **{sequential.body[3]: net.features[0] for sequential in net.features[2]},  # stage1
        **{sequential.body[3]: net.features[3][0].downsample[0] for sequential in net.features[3]},     # stage2
        **{sequential.body[3]: net.features[4][0].downsample[0] for sequential in net.features[4]}      # stage3
    }

    return convs, bns, shares
