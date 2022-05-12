import numpy as np

import paddle
from paddle import sparse
import time

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
class SpMiddleFHD(paddle.nn.Layer):
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
        self.middle_conv = paddle.nn.Sequential(
            sparse.SubmConv3D(num_input_features, 16, 3), 
            sparse.BatchNorm(16, epsilon=1e-3, momentum=0.01),
            sparse.ReLU(),
            sparse.SubmConv3D(16, 16, 3),
            sparse.BatchNorm(16, epsilon=1e-3, momentum=0.01),
            sparse.ReLU(),
            sparse.Conv3D(16, 32, 3, 2, padding=1),  # [1600, 1200, 41] -> [800, 600, 21]
            sparse.BatchNorm(32, epsilon=1e-3, momentum=0.01),
            sparse.ReLU(),
            sparse.SubmConv3D(32, 32, 3),
            sparse.BatchNorm(32, epsilon=1e-3, momentum=0.01),
            sparse.ReLU(),
            sparse.SubmConv3D(32, 32, 3),
            sparse.BatchNorm(32, epsilon=1e-3, momentum=0.01),
            sparse.ReLU(),
            sparse.Conv3D(32, 64, 3, 2, padding=1),  # [800, 600, 21] -> [400, 300, 11]
            sparse.BatchNorm(64, epsilon=1e-3, momentum=0.01),
            sparse.ReLU(),
            sparse.SubmConv3D(64, 64, 3),
            sparse.BatchNorm(64, epsilon=1e-3, momentum=0.01),
            sparse.ReLU(),
            sparse.SubmConv3D(64, 64, 3),
            sparse.BatchNorm(64, epsilon=1e-3, momentum=0.01),
            sparse.ReLU(),
            sparse.SubmConv3D(64, 64, 3),
            sparse.BatchNorm(64, epsilon=1e-3, momentum=0.01),
            sparse.ReLU(),
            sparse.Conv3D(64, 64, 3, 2, padding=[0, 1, 1]),  # [400, 300, 11] -> [200, 150, 5]
            sparse.BatchNorm(64, epsilon=1e-3, momentum=0.01),
            sparse.ReLU(),
            sparse.SubmConv3D(64, 64, 3),
            sparse.BatchNorm(64, epsilon=1e-3, momentum=0.01),
            sparse.ReLU(),
            sparse.SubmConv3D(64, 64, 3),
            sparse.BatchNorm(64, epsilon=1e-3, momentum=0.01),
            sparse.ReLU(),
            sparse.SubmConv3D(64, 64, 3), 
            sparse.BatchNorm(64, epsilon=1e-3, momentum=0.01),
            sparse.ReLU(),
            sparse.Conv3D(64, 64, (3, 1, 1), (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            sparse.BatchNorm(64, epsilon=1e-3, momentum=0.01),
            sparse.ReLU(),
        )
        self.max_batch_size = 6
        # self.grid = torch.full([self.max_batch_size, *sparse_shape], -1, dtype=torch.int32).cuda()

    def forward(self, voxel_features, coors, batch_size, shape):
        # coors[:, 1] += 1
        #coors = coors.int()
        #ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
        #                              batch_size)
        #ret = self.middle_conv(ret)
        #ret = ret.dense()

        #N, C, D, H, W = ret.shape
        #ret = ret.view(N, C * D, H, W)
        #return ret

        coors = paddle.cast(coors, 'int64')
        sp_x = sparse.sparse_coo_tensor(coors, voxel_features, shape)
        out = self.middle_conv(sp_x)
        out = out.to_dense()
        out = paddle.transpose(out, perm=[0, 4, 1, 2, 3])
        N, C, D, H, W = out.shape
        out = paddle.reshape(out, shape=[N, C*D, H, W])
        return out

from paddle.fluid.framework import _test_eager_guard
import paddle


#sparse_shape =  [  41 1600 1408]
#output_shape =  [1, 40, 1600, 1408, 16]
#coors.shape= torch.Size([136000, 4])
#features.shape= torch.Size([136000, 4])
#batch_size= 8
def test():
    with _test_eager_guard():
        batch_size = 8
        in_channels = 4 
        in_shape = [41, 1600, 1408]
        full_shape = [batch_size] + in_shape + [in_channels]
        out_shape = [8, 40, 1600, 1408, 16]
        model = SpMiddleFHD(out_shape, num_input_features = in_channels)
        nnz = 13600
        features = paddle.randn((nnz, in_channels)) 
        print(features.shape)

        coors = []
        for i in range(4):
           coors.append(paddle.randint(0, full_shape[i], [1, nnz])) 
        coors = paddle.concat(coors)
        print(coors.shape)

        a = paddle.sparse.sparse_coo_tensor(coors, features)
        np.save("features", a.values().numpy())
        np.save("coors", np.transpose(a.indices().numpy(), (1, 0)))

        t0 = time.time()
        out = model(a.values(), a.indices(), batch_size, full_shape)
        t1 = time.time()
        print("time = ", t1-t0)
        print("out.shape", out.shape)

        np.save("out", out.numpy())

#test()
