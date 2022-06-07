import numpy as np

import paddle
from paddle.incubate import sparse
from paddle.incubate.sparse import nn 
import time

REGISTERED_MIDDLE_CLASSES = {}
debug = 0

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

flag = 0

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
        self.sparse_shape = sparse_shape
        self.num_input_features  = num_input_features
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        print("num_input_features=", num_input_features)
        self.middle_conv = paddle.nn.Sequential(
            nn.SubmConv3D(num_input_features, 16, 3), 
            #nn.BatchNorm(16, epsilon=1e-3, momentum=0.01),
            nn.ReLU(),

            nn.SubmConv3D(16, 16, 3),
            #nn.BatchNorm(16, epsilon=1e-3, momentum=0.01),
            nn.ReLU(),

            nn.Conv3D(16, 32, 3, 2, padding=1),  # [1600, 1200, 41] -> [800, 600, 21]
            #nn.BatchNorm(32, epsilon=1e-3, momentum=0.01),
            nn.ReLU(),

            nn.SubmConv3D(32, 32, 3),
            #nn.BatchNorm(32, epsilon=1e-3, momentum=0.01),
            nn.ReLU(),

            nn.SubmConv3D(32, 32, 3),
            #nn.BatchNorm(32, epsilon=1e-3, momentum=0.01),
            nn.ReLU(),

            nn.Conv3D(32, 64, 3, 2, padding=1),  # [800, 600, 21] -> [400, 300, 11]
            #nn.BatchNorm(64, epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.SubmConv3D(64, 64, 3),
            #nn.BatchNorm(64, epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.SubmConv3D(64, 64, 3),
            #nn.BatchNorm(64, epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.SubmConv3D(64, 64, 3),
            #nn.BatchNorm(64, epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv3D(64, 64, 3, 2, padding=[0, 1, 1]),  # [400, 300, 11] -> [200, 150, 5]
            #nn.BatchNorm(64, epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.SubmConv3D(64, 64, 3),
            #nn.BatchNorm(64, epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.SubmConv3D(64, 64, 3),
            #nn.BatchNorm(64, epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.SubmConv3D(64, 64, 3), 
            #nn.BatchNorm(64, epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv3D(64, 64, (3, 1, 1), (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            #nn.BatchNorm(64, epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.max_batch_size = 6
        # self.grid = torch.full([self.max_batch_size, *sparse_shape], -1, dtype=torch.int32).cuda()

        #for debug
        if debug:
            for i in range(int(len(self.middle_conv)/2)):
                weight = np.load("torch_middle_weight" + str(i) + ".npy")
                self.middle_conv[i*2].weight.set_value(paddle.to_tensor(weight, stop_gradient=True))

    def forward(self, voxel_features, coors, batch_size):
        shape = [batch_size] + list(self.sparse_shape) + [self.num_input_features]
        sp_x = sparse.sparse_coo_tensor(coors.transpose((1,0)), voxel_features, shape=shape, stop_gradient=False)
        out = self.middle_conv(sp_x)

        #out = out.to_dense()
        out = sparse.sparse_coo_tensor(paddle.cast(out.indices(), 'int64'), out.values(), out.shape, stop_gradient=False)
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
        batch_size = 1
        in_channels = 4 
        in_shape = [41, 160, 140]
        full_shape = [batch_size] + in_shape + [in_channels]
        out_shape = [8, 40, 160, 140, 16]
        model = SpMiddleFHD(out_shape, num_input_features = in_channels)
        nnz = 13600
        features = paddle.randn((nnz, in_channels)) 
        print(features.shape)

        coors = []
        for i in range(4):
           coors.append(paddle.randint(0, full_shape[i], [1, nnz])) 
        coors = paddle.concat(coors)
        print(coors.shape)

        a = sparse.sparse_coo_tensor(coors, features, stop_gradient=False)
        np.save("features", a.values().numpy())
        np.save("coors", np.transpose(a.indices().numpy(), (1, 0)))

        t0 = time.time()
        out = model(a.values(), a.indices().transpose((1,0)), batch_size)
        t1 = time.time()
        print("time = ", t1-t0)
        print("out.shape", out.shape)
        conv = paddle.nn.Conv2D(128, 64, 3)
        out2 = conv(out)
        out2.backward(out2)
        print(a.grad.shape)
        print(model.middle_conv[0].weight.grad.shape)

        np.save("out", out.numpy())

#test()

import time
def test_by_real_data():
    in_coors = np.load('in_coors.npy') 
    in_features = np.load('in_features.npy') 
    in_channels = in_features.shape[1]
    batch_size = 8
    sparse_shape =  [41, 1600, 1408]
    output_shape =  [1, 40, 1600, 1408, 16]
    full_shape = [batch_size] + sparse_shape + [in_channels]
    with _test_eager_guard():
        model = SpMiddleFHD(output_shape, num_input_features = in_channels)
        indices = paddle.to_tensor(in_coors)
        values = paddle.to_tensor(in_features)
        print(indices.shape, values.shape)
        out = model(values, indices, batch_size)
        print(out.shape)

        t0 = time.time()
        for i in range(100):
            out = model(values, indices, batch_size)
        t1 = time.time()
        print("paddle time:", t1-t0)


#test_by_real_data()
