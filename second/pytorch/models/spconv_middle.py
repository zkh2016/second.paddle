import time

import numpy as np
import spconv
#import spconv.pytorch as spconv
import torch
from torch import nn
from torch.nn import functional as F

REGISTERED_MIDDLE_CLASSES = {}

def register_middle(cls, name=None):
    global REGISTERED_MIDDLE_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MIDDLE_CLASSES, f"exist class: {REGISTERED_MIDDLE_CLASSES}"
    REGISTERED_MIDDLE_CLASSES[name] = cls
    return cls

def get_middle_class(name):
    global REGISTERED_MIDDLE_CLASSES
    assert name in REGISTERED_MIDDLE_CLASSES, f"available class: {REGISTERED_MIDDLE_CLASSES}"
    return REGISTERED_MIDDLE_CLASSES[name]


@register_middle
class SpMiddleFHD(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleFHD'):
        super(SpMiddleFHD, self).__init__()
        self.name = name
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            spconv.SubMConv3d(num_input_features, 16, 3, indice_key="subm0", bias=False),
            #nn.BatchNorm1d(16, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(16, 16, 3, indice_key="subm0", bias=False),
            #nn.BatchNorm1d(16, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SparseConv3d(16, 32, 3, 2,
                     padding=1, bias=False),  # [1600, 1200, 41] -> [800, 600, 21]
            #nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            #spconv.SubMConv3d(32, 32, 3, indice_key="subm1", bias=False),
            ##nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
            #nn.ReLU(),
            #spconv.SubMConv3d(32, 32, 3, indice_key="subm1", bias=False),
            ##nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
            #nn.ReLU(),
            #spconv.SparseConv3d(32, 64, 3, 2,
            #         padding=1, bias=False),  # [800, 600, 21] -> [400, 300, 11]
            ##nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            #nn.ReLU(),
            #spconv.SubMConv3d(64, 64, 3, indice_key="subm2", bias=False),
            ##nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            #nn.ReLU(),
            #spconv.SubMConv3d(64, 64, 3, indice_key="subm2", bias=False),
            ##nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            #nn.ReLU(),
            #spconv.SubMConv3d(64, 64, 3, indice_key="subm2", bias=False),
            ##nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            #nn.ReLU(),
            #spconv.SparseConv3d(64, 64, 3, 2,
            #         padding=[0, 1, 1], bias=False),  # [400, 300, 11] -> [200, 150, 5]
            ##nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            #nn.ReLU(),
            #spconv.SubMConv3d(64, 64, 3, indice_key="subm3", bias=False),
            ##nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            #nn.ReLU(),
            #spconv.SubMConv3d(64, 64, 3, indice_key="subm3", bias=False),
            ##nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            #nn.ReLU(),
            #spconv.SubMConv3d(64, 64, 3, indice_key="subm3", bias=False),
            ##nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            #nn.ReLU(),
            #spconv.SparseConv3d(64, 64, (3, 1, 1),
            #         (2, 1, 1), bias=False),  # [200, 150, 5] -> [200, 150, 2]
            ##nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            #nn.ReLU(),
        )
        self.max_batch_size = 6
        # self.grid = torch.full([self.max_batch_size, *sparse_shape], -1, dtype=torch.int32).cuda()

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        # t = time.time()
        # torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        # torch.cuda.synchronize()
        # print("spconv forward time", time.time() - t)
        #ret = ret.dense()

        #N, C, D, H, W = ret.shape
        #ret = ret.view(N, C * D, H, W)
        return ret


#sparse_shape =  [  41 1600 1408]
#output_shape =  [1, 40, 1600, 1408, 16]
#coors.shape= torch.Size([136000, 4])
#features.shape= torch.Size([136000, 4])
#batch_size= 8

def test():
    batch_size = 8
    in_channels = 4 
    out_shape = [8, 40, 1600, 1408, 16]
    model = SpMiddleFHD(out_shape, num_input_features = in_channels)

    features = np.load("features.npy")
    coors = np.load("coors.npy")

    device = torch.device("cuda")
    features = torch.tensor(features, device=device)
    coors = torch.tensor(coors, device=device)

    out = model(features, coors, batch_size)

    ref_out = np.load("out.npy")
    assert np.allclose(out.numpy(), ref_out, atol=1e-3)

#test()

def test_by_real_data():
    in_coors = np.load('in_coors.npy') 
    in_features = np.load('in_features.npy') 
    in_channels = in_features.shape[1]
    batch_size = 8
    sparse_shape =  [41, 1600, 1408]
    output_shape =  [1, 40, 1600, 1408, 16]
    full_shape = [batch_size] + sparse_shape + [in_channels]
    device = torch.device('cuda')

    model = SpMiddleFHD(output_shape, num_input_features = in_channels)
    model.to(device)
    indices = torch.tensor(in_coors, device=device)
    values = torch.tensor(in_features, device=device)
    print(indices.size(), values.size())
    out = model(values, indices, batch_size)
    #print(out.shape)

    t0 = time.time()
    for i in range(100):
        out = model(values, indices, batch_size)
    t1 = time.time()
    print("torch time:", t1-t0)


test_by_real_data()
