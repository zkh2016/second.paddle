#import torch
import paddle
import numpy as np

def one_hot(tensor, depth, dim=-1, on_value=1.0, dtype=paddle.float32):
    #tensor_onehot = torch.zeros(
    #    *list(tensor.shape), depth, dtype=dtype, device=tensor.device)
    #tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)
    target_shape = tensor.shape + [depth]
    tensor_onehot = paddle.zeros(target_shape, dtype)
    #tensor_onehot.scatter_(paddle.cast(tensor.unsqueeze(dim), dtype='int64'), on_value)
    #return tensor_onehot
    #scatter_(tensor_onehot, paddle.cast(tensor.unsqueeze(dim), dtype='int64'), on_value, dim)
    tensor_onehot.put_along_axis_(paddle.cast(tensor.unsqueeze(dim), dtype='int64'), on_value, dim)
    return tensor_onehot
