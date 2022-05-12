#import torch
import paddle

def one_hot(tensor, depth, dim=-1, on_value=1.0, dtype=paddle.float32):
    #tensor_onehot = torch.zeros(
    #    *list(tensor.shape), depth, dtype=dtype, device=tensor.device)
    #tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)
    tensor_onehot = paddle.zeros(
        *list(tensor.shape), depth, dtype=dtype)
    paddle.scatter(tensor_onehot, paddle.cast(tensor.unsqueeze(dim), dtype='int64'), on_value)
    return tensor_onehot
