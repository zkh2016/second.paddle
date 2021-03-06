import numpy as np
import paddle

REGISTERED_VFE_CLASSES = {}

def register_vfe(cls, name=None):
    global REGISTERED_VFE_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_VFE_CLASSES, f"exist class: {REGISTERED_VFE_CLASSES}"
    REGISTERED_VFE_CLASSES[name] = cls
    return cls

def get_vfe_class(name):
    global REGISTERED_VFE_CLASSES
    assert name in REGISTERED_VFE_CLASSES, f"available class: {REGISTERED_VFE_CLASSES}"
    return REGISTERED_VFE_CLASSES[name]


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = paddle.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = paddle.arange(
        max_num, dtype='int32').reshape(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = paddle.cast(actual_num, 'int32') > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator

class VFELayer(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, use_norm=True, name='vfe'):
        super(VFELayer, self).__init__()
        self.name = name
        self.units = int(out_channels / 2)
        self.linear = paddle.nn.Linear(in_channels, self.units, bias_attr=False)
        self.norm = paddle.nn.BatchNorm1D(self.units, epsilon=1e-3, momentum=0.01)

    def forward(self, inputs):
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        voxel_count = inputs.shape[1]
        x = self.linear(inputs)
        x = self.norm(x.transpose((0, 2, 1))).transpose((0, 2, 1))
        pointwise = paddle.nn.funtional.relu(x)
        # [K, T, units]

        aggregated = paddle.max(pointwise, axis=1, keepdim=True)[0]
        # [K, 1, units]
        #repeated = aggregated.repeat(1, voxel_count, 1)
        repeated = paddle.concat([aggregated, aggregated], axis=1)

        #concatenated = torch.cat([pointwise, repeated], dim=2)
        concatenated = paddle.concat([pointwise, repeated], axis=2)
        # [K, T, 2 * units]
        return concatenated

@register_vfe
class VoxelFeatureExtractor(paddle.nn.Layer):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 name='VoxelFeatureExtractor'):
        super(VoxelFeatureExtractor, self).__init__()
        self.name = name
        assert len(num_filters) == 2
        num_input_features += 3  # add mean features
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
        self.vfe1 = VFELayer(num_input_features, num_filters[0], use_norm)
        self.vfe2 = VFELayer(num_filters[0], num_filters[1], use_norm)
        self.linear = paddle.nn.Linear(num_filters[1], num_filters[1], bias_attr=False)
        # var_torch_init(self.linear.weight)
        # var_torch_init(self.linear.bias)
        self.norm = paddle.nn.BatchNorm1D(num_filters[1], epsilon=1e-3, momentum=0.01)

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        #points_mean = features[:, :, :3].sum(
        #    dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        points_mean = features[:, :, :3].sum(axis=1, keepdim=True) / paddle.cast(num_voxels, features.dtype).reshape((-1, 1, 1))
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            #points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            points_dist = paddle.linalg.norm(features[:, :, :3], p=2, axis=2, keepdim=True)
            features = paddle.concat([features, features_relative, points_dist],
                                 axis=-1)
        else:
            features = paddle.concat([features, features_relative], axis=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = paddle.cast(paddle.unsqueeze(mask, -1), features.dtype)
        # mask = features.max(dim=2, keepdim=True)[0] != 0
        x = self.vfe1(features)
        x *= mask
        x = self.vfe2(x)
        x *= mask
        x = self.linear(x)
        x = self.norm(x.transpose((0, 2, 1))).transpose((0, 2, 1))
        x = F.relu(x)
        x *= mask
        # x: [concated_num_points, num_voxel_size, 128]
        voxelwise = paddle.max(x, axis=1)[0]
        return voxelwise

@register_vfe
class VoxelFeatureExtractorV2(paddle.nn.Layer):
    """VoxelFeatureExtractor with arbitrary number of VFE. deprecated.
    """
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 name='VoxelFeatureExtractor'):
        super(VoxelFeatureExtractorV2, self).__init__()
        self.name = name
        assert len(num_filters) > 0
        num_input_features += 3
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        num_filters = [num_input_features] + num_filters
        filters_pairs = [[num_filters[i], num_filters[i + 1]]
                         for i in range(len(num_filters) - 1)]
        #self.vfe_layers = nn.ModuleList(
        self.vfe_layers = paddle.nn.LayerList(
            [VFELayer(i, o, use_norm) for i, o in filters_pairs])
        self.linear = paddle.nn.Linear(num_filters[-1], num_filters[-1], bias_attr=False)
        # var_torch_init(self.linear.weight)
        # var_torch_init(self.linear.bias)
        self.norm = paddle.nn.BatchNorm1D(num_filters[-1], epsilon=1e-3, momentum=0.01)

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :3].sum(
            axis=1, keepdim=True) / paddle.cast(num_voxels, features.dtype).reshape((-1, 1, 1))
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_dist = paddle.linalg.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = paddle.concat([features, features_relative, points_dist],
                                 axis=-1)
        else:
            features = paddle.concat([features, features_relative], axis=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = paddle.cast(paddle.unsqueeze(mask, -1), features.dtype)
        for vfe in self.vfe_layers:
            features = vfe(features)
            features *= mask
        features = self.linear(features)
        features = self.norm(features.transpose(0, 2, 1)).transpose((0, 2, 1))
        features = paddle.nn.functional.relu(features)
        features *= mask
        # x: [concated_num_points, num_voxel_size, 128]
        voxelwise = paddle.max(features, axis=1)[0]
        return voxelwise

@register_vfe
class SimpleVoxel(paddle.nn.Layer):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 name='VoxelFeatureExtractor'):
        super(SimpleVoxel, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :self.num_input_features].sum(
            axis=1, keepdim=False) / paddle.cast(num_voxels, features.dtype).reshape((-1, 1))
        return points_mean

@register_vfe
class SimpleVoxelRadius(paddle.nn.Layer):
    """Simple voxel encoder. only keep r, z and reflection feature.
    """

    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(32, 128),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 name='SimpleVoxelRadius'):

        super(SimpleVoxelRadius, self).__init__()

        self.num_input_features = num_input_features
        self.name = name

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :self.num_input_features].sum(
            axis=1, keepdim=False) / paddle.cast(num_voxels, features.dtype).reshape((-1, 1))
        feature = paddle.linalg.norm(points_mean[:, :2], p=2, axis=1, keepdim=True)
        # z is important for z position regression, but x, y is not.
        res = paddle.concat([feature, points_mean[:, 2:self.num_input_features]],
                        axis=1)
        return res

def test():
    in_channels = 4
    vfe = SimpleVoxel(num_input_features=in_channels)
    
    concate_num_points = 100
    num_voxel_size = 100
    features_shape = [concate_num_points, num_voxel_size, in_channels]
    num_voxels_shape =  [concate_num_points]

    features = paddle.randn(features_shape)
    num_voxels = paddle.randn(num_voxels_shape)

    coors = paddle.randint(4, concate_num_points)
    out = vfe(features, num_voxels, coors)

#test()
