import time

import numpy as np
#import spconv
import spconv.pytorch as spconv
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
            nn.BatchNorm1d(16, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(16, 16, 3, indice_key="subm0", bias=False),
            nn.BatchNorm1d(16, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SparseConv3d(16, 32, 3, 2,
                     padding=1, bias=False),  # [1600, 1200, 41] -> [800, 600, 21]
            nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(32, 32, 3, indice_key="subm1", bias=False),
            nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(32, 32, 3, indice_key="subm1", bias=False),
            nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SparseConv3d(32, 64, 3, 2,
                     padding=1, bias=False),  # [800, 600, 21] -> [400, 300, 11]
            nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(64, 64, 3, indice_key="subm2", bias=False),
            nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(64, 64, 3, indice_key="subm2", bias=False),
            nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(64, 64, 3, indice_key="subm2", bias=False),
            nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SparseConv3d(64, 64, 3, 2,
                     padding=[0, 1, 1], bias=False),  # [400, 300, 11] -> [200, 150, 5]
            nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(64, 64, 3, indice_key="subm3", bias=False),
            nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(64, 64, 3, indice_key="subm3", bias=False),
            nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(64, 64, 3, indice_key="subm3", bias=False),
            nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SparseConv3d(64, 64, (3, 1, 1),
                     (2, 1, 1), bias=False),  # [200, 150, 5] -> [200, 150, 2]
            nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
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
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret

@register_middle
class SpMiddleFHDPeople(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleFHD'):
        super(SpMiddleFHDPeople, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SubMConv3d(16, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SpConv3d(16, 32, 3, 2,
                     padding=1),  # [1600, 1200, 21] -> [800, 600, 11]
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2,
                     padding=[0, 1, 1]),  # [800, 600, 11] -> [400, 300, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1),
                     (2, 1, 1)),  # [400, 300, 5] -> [400, 300, 2]
            BatchNorm1d(64),
            nn.ReLU(),
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
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret

@register_middle
class SpMiddle2K(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddle2K'):
        super(SpMiddle2K, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(
                num_input_features, 8, 3,
                indice_key="subm0"),  # [3200, 2400, 81] -> [1600, 1200, 41]
            BatchNorm1d(8),
            nn.ReLU(),
            SubMConv3d(8, 8, 3, indice_key="subm0"),
            BatchNorm1d(8),
            nn.ReLU(),
            SpConv3d(8, 16, 3, 2,
                     padding=1),  # [1600, 1200, 41] -> [800, 600, 21]
            BatchNorm1d(16),
            nn.ReLU(),
            SubMConv3d(16, 16, 3, indice_key="subm1"),
            BatchNorm1d(16),
            nn.ReLU(),
            SubMConv3d(16, 16, 3, indice_key="subm1"),
            BatchNorm1d(16),
            nn.ReLU(),
            SpConv3d(16, 32, 3, 2,
                     padding=1),  # [1600, 1200, 41] -> [800, 600, 21]
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm2"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm2"),
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2,
                     padding=1),  # [800, 600, 21] -> [400, 300, 11]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, 3, 2,
                     padding=[0, 1, 1]),  # [400, 300, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm4"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm4"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm4"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1),
                     (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            nn.ReLU(),
        )
        self.max_batch_size = 3
        self.grid = torch.full([self.max_batch_size, *sparse_shape],
                               -1,
                               dtype=torch.int32).cuda()

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size, self.grid)
        # t = time.time()
        # torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        # torch.cuda.synchronize()
        # print("spconv forward time", time.time() - t)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret

@register_middle
class SpMiddleFHDLite(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleFHDLite'):
        super(SpMiddleFHDLite, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SpConv3d(num_input_features, 16, 3, 2,
                     padding=1),  # [1600, 1200, 41] -> [800, 600, 21]
            BatchNorm1d(16),
            nn.ReLU(),
            SpConv3d(16, 32, 3, 2,
                     padding=1),  # [800, 600, 21] -> [400, 300, 11]
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2,
                     padding=[0, 1, 1]),  # [400, 300, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1),
                     (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.middle_conv(ret)

        # ret.features = F.relu(ret.features)
        # print(self.middle_conv.fused())
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret

@register_middle
class SpMiddleFHDLiteHRZ(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleFHDLite'):
        super(SpMiddleFHDLiteHRZ, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SpConv3d(num_input_features, 32, 3, 2,
                     padding=1),  # [1600, 1200, 81] -> [800, 600, 41]
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2,
                     padding=1),  # [800, 600, 41] -> [400, 300, 21]
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, 3, 2,
                     padding=1),  # [400, 300, 21] -> [200, 150, 11]
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1),
                     (2, 1, 1)),  # [200, 150, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1),
                     (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret

@register_middle
class SpMiddleFHDHRZ(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleFHD'):
        super(SpMiddleFHDHRZ, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
        else:
            BatchNorm1d = Empty
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SubMConv3d(16, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SpConv3d(16, 32, 3, 2,
                     padding=1),  # [1600, 1200, 81] -> [800, 600, 41]
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2,
                     padding=1),  # [800, 600, 41] -> [400, 300, 21]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, 3, 2,
                     padding=1),  # [400, 300, 21] -> [200, 150, 11]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1),
                     (2, 1, 1)),  # [200, 150, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm4"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm4"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1),
                     (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int().contiguous()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
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

test()
