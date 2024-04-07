# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Gradient compression algorithms."""

import torch
from torch.autograd import Function
from math import floor



def q_bit_com_old(tensor, q, dim=-1):
    try:
        max_num = torch.max(tensor, dim=dim, keepdim=True).values
    except:
        max_num = 0
    try:
        min_num = torch.min(tensor, dim=dim, keepdim=True).values
    except:
        min_num = 0
    s = (max_num - min_num) / (pow(2, q) - 1)
    if isinstance(s,int) and s==0:
        return tensor, 1, 0
    z = - (min_num / (s))
    int_tensor = torch.round(tensor / (s) + z)
    return int_tensor, s, z


def q_bit_com(tensor, q, dim=-1):
    if tensor.shape[0] == 0:
        return tensor, 1, 0
    max_num = torch.max(tensor, dim=dim, keepdim=True).values
    min_num = torch.min(tensor, dim=dim, keepdim=True).values
    s = (max_num - min_num) / (pow(2, q) - 1)
    z = - (min_num / (s))
    int_tensor = torch.round(tensor / (s) + z)
    return int_tensor, s, z


def q_bit_dec(tensor, dim=-1):
    int_tensor, s, z = tensor
    restore_tensor = s * (int_tensor - z)
    return restore_tensor

def shapshft(tensor, q, dim=-1):
    compressed_tensor = q_bit_com(tensor, q, dim=dim)
    restore_tensor = q_bit_dec(compressed_tensor, dim=dim)
    return restore_tensor


def pwlq(x, q):
    tensor = x.flatten()
    dist = "norm"
    tensor_std = torch.std(tensor) + 1e-12
    tensor_abs = torch.abs(tensor)
    try:
        abs_max = torch.max(tensor_abs, dim=-1, keepdim=True).values
    except:
        abs_max = 1e-5
    abs_max_normalized = abs_max / tensor_std
    if dist == 'norm':
        break_point_normalized = torch.log(0.86143114 * abs_max_normalized + 0.607901097496529 )
    elif dist == 'laplace':
        break_point_normalized = 0.80304483 * torch.sqrt(abs_max_normalized) - 0.3166785508381478
    bkp_ratio = break_point_normalized / abs_max_normalized
    break_point = bkp_ratio * abs_max

    neg_idx = (tensor < -break_point)
    pos_idx = (tensor > break_point)
    mid_idx = torch.logical_not(torch.logical_or(neg_idx, pos_idx))
    res_neg = q_bit_com(tensor[neg_idx], q-1)
    res_pos = q_bit_com(tensor[pos_idx], q-1)
    res_mid = q_bit_com(tensor[mid_idx], q)

    mid_idx = torch.logical_not(torch.logical_or(neg_idx, pos_idx))

    restore_tensor = torch.zeros_like(tensor)
    restore_tensor[neg_idx] = q_bit_dec(res_neg)
    restore_tensor[pos_idx] = q_bit_dec(res_pos)
    restore_tensor[mid_idx] = q_bit_dec(res_mid)
    return restore_tensor.reshape(x.shape)



def tg(x):
    x_1d = x.view(-1)
    max_ = torch.max(torch.abs(x_1d))
    p_i = torch.clamp(torch.abs(x_1d) / max_, min=0.00000001, max=0.9999999)
    idx = torch.where((torch.rand(len(x_1d), device = x.device) < p_i) == True)[0]
    value = torch.sign(x_1d[idx]) * max_
    out = torch.zeros(len(x_1d),device=x.device)
    out[idx] = value
    return out.reshape(x.shape)


def lq(x,q):
    layer_1d = x.view(-1)
    s = (torch.max(layer_1d) - torch.min(layer_1d)) / (torch.pow(torch.tensor(2), q) - 1)
    z = - (torch.min(layer_1d) / (s + 0.0001*s))
    int_layer_1d = torch.round(layer_1d / (s + 0.0001*s) + z)
    restore_layer = s * (int_layer_1d - z)
    return restore_layer.reshape(x.shape)



def topk(x,sr):
    x_1d = x.view(-1)
    dev = torch.device(x.device)
    _, idx = torch.abs(x_1d).topk(floor(sr * len(x_1d)))
    out = torch.zeros(len(x_1d),device=dev)
    out[idx] = x_1d[idx]
    #x_1d.scatter_(0,idx,0)
    return out.reshape(x.shape)#,x_1d


def rq(x,q):
    layer_1d = x.view(-1)
    a = torch.abs(layer_1d)/torch.max(torch.abs(layer_1d))
    s = torch.pow(torch.tensor(2), q)
    l = torch.floor(a/(1/s))
    p_ceil = a*s-l
    idx = torch.where((torch.rand(len(p_ceil), device = torch.device(layer_1d.device)) < p_ceil) == True)[0]
    l[idx]+=1
    restore_layer = torch.max(torch.abs(layer_1d))*torch.sign(layer_1d)*l/s
    return restore_layer.reshape(x.shape)



class disp_comp(Function):
    @staticmethod
    def forward(ctx,x, q, id):
        # x_restore = rq(x, q) # ours
        # x_restore = lq(x, q)
        #x_restore = pwlq(x, q)
        x_restore = shapshft(x, q, dim=-1)
        return x_restore

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None, None


class comb_comp(Function):
    @staticmethod
    def forward(ctx, x, q):
        # x_restore = rq(x, q) # ours
        # x_restore = lq(x, q)
        #x_restore = pwlq(x, q)
        x_restore = shapshft(x, q, dim=-1)
        return x_restore

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None



class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""
    @staticmethod
    def compress(tensor):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        pass

    @staticmethod
    def decompress(tensor, ctx):
        """Decompress the tensor with the given context."""
        pass


class NoneCompressor(Compressor):
    """Default no-op compression."""
    @staticmethod
    def compress(tensor):
        """Returns the tensor unmodified."""
        print(tensor.size())
        return tensor, None

    @staticmethod
    def decompress(tensor, ctx):
        print(tensor.size())
        """Returns the tensor unmodified."""
        return tensor


class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""
    @staticmethod
    def compress(tensor):
        """Downcasts the tensor to 16-bit."""
        tensor_compressed = tensor
        if tensor.dtype.is_floating_point:
            # Only allow compression from other floating point types
            tensor_compressed = tensor.type(torch.float16)
        return tensor_compressed, tensor.dtype

    @staticmethod
    def decompress(tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating_point:
            tensor_decompressed = tensor.type(dtype)
        return tensor_decompressed

class TOPKCompressor(Compressor):
    @staticmethod
    def compress(tensor):
        tensor_shape = tensor.size()
#        print('compress shape',tensor_shape)
        tensor_1d = tensor.flatten()
        k = max(1, int(0.1 * tensor_1d.size(0)))
        _, index = torch.topk(tensor_1d.abs(), k)
        result = torch.zeros(tensor_1d.size()).cuda()
        result[index] = tensor_1d[index]
        result = result.reshape(tensor_shape)
#        ctx = [index, tensor_1d[index], tensor_shape, tensor_1d.size()]
#        print('compress index',index)
#        return torch.zeros(1).cuda(), ctx
#         print('compress', result.size(), tensor_shape, result.device)
        return result, [tensor_shape]
    @staticmethod
    def decompress(tensor, ctx):
#        index, value, tensor_shape, tensor_len = ctx
#        print('decompress shape',tensor_shape)
#        print('decompress index',index)
#        result_1d = torch.zeros(tensor_len, device=tensor.device)
#        result_1d[index] = value
#        result = result_1d.reshape(*tensor_shape)
#        print('decompress', ctx[0].size(), ctx[1], tensor.device)
#        result = ctx[0].reshape(*ctx[1])
       return tensor

class Compression(object):
    """Optional gradient compression algorithm used during allreduce."""

    """Do not compress the gradients. This is the default."""
    none = NoneCompressor

    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor


    topk = TOPKCompressor
