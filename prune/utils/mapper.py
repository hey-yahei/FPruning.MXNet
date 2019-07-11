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

import mxnet as mx

__all__ = ['CrossMapper', 'get_gluon_symbol_mapper', 'get_conv_bn_pairs']
__author__ = 'YaHei'


class CrossMapper(object):
    """ Cross mapper for A and B """
    def __init__(self, **kwargs):
        """
        Accept two keyword-arguments with list-type value.
        Note that the length of lists should be euivalent.
        example:
            > mapper = CrossMapper(upper=["A","B","C"], lower=["a","b","c"])
            > print(mapper.upper_list)
                ["A","B","C"]
            > print(mapper.lower_list)
                ["a","b","c"]
            > mapper.get_upper("a")
                "A"
            > mapper.get_lower("A")
                "a"
        """
        assert len(kwargs) == 2, f'len(kwargs) == 2, ({len(kwargs)} vs 2)'
        (name1, list1), (name2, list2) = kwargs.items()
        self._list1 = list1
        self._list2 = list2

        self.__setattr__(f"{name1}_list", self._list1)
        self.__setattr__(f"{name2}_list", self._list2)
        self.__setattr__(f"get_{name1}", self._get_from_list2)
        self.__setattr__(f"get_{name2}", self._get_from_list1)

    def _get_from_list1(self, obj):
        idx = self._list1.index(obj)
        return self._list2[idx]

    def _get_from_list2(self, obj):
        idx = self._list2.index(obj)
        return self._list1[idx]


def get_gluon_symbol_mapper(net):
    """
    Get a cross-mapper for gluon instance and names of symbol.
    :param net: mxnet.gluon.nn.HybridBlock
        The gluon net.
    :return: CrossMapper
        A cross-mapper for gluon instance and names of symbol.
    """
    # Collect argument names and auxiliary_state names
    out = net(mx.sym.var("data"))
    arg_aux = out.list_arguments() + out.list_auxiliary_states()
    # Collect gluon and symbol name if gluon and symbol is 1-1 correspondence
    gluon_list = []
    symbol_name_list = []
    def _collect_symbol_names(m):
        out_ = m(mx.sym.var("data"))
        symbols = [s for s in out_.get_internals() if s.name not in arg_aux]
        if len(symbols) == 1:
            gluon_list.append(m)
            symbol_name_list.append(symbols[0].name)
    _ = net.apply(_collect_symbol_names)

    return CrossMapper(gluon=gluon_list, symbol_name=symbol_name_list)


def get_conv_bn_pairs(net):
    """
    Get a cross-mapper for convolution block and batchnorm block.
    :param net: mxnet.gluon.nn.HybridBlock
        The gluon net.
    :return: CrossMapper
        A cross-mapper for convolution block and batchnorm block.
    """
    gs_mapper = get_gluon_symbol_mapper(net)
    out = net(mx.sym.var("data"))
    # Collect convolution-bn pairs
    conv_list = []
    bn_list = []
    def _collect_conv_bn(m):
        # Deal with BatchNorm blocks
        if isinstance(m, mx.gluon.nn.BatchNorm):
            # Get bn symbol
            bn_sym_name = gs_mapper.get_symbol_name(m)
            bn_sym = out.get_internals()[f'{bn_sym_name}_output']
            # Check the bottom block of bn
            for child in bn_sym.get_children():
                child_gluon = gs_mapper.get_gluon(child.name)
                # If the bottom of BatchNorm is Convolution, collect them to list
                if child_gluon is not None and isinstance(child_gluon, mx.gluon.nn.Conv2D):
                    conv_list.append(child_gluon)
                    bn_list.append(m)
                    break
    _ = net.apply(_collect_conv_bn)

    return CrossMapper(conv=conv_list, bn=bn_list)


