import numpy as np

import paddle
from paddle import nn
from paddle.nn import functional as F
from .common import Sequential

REGISTERED_RPN_CLASSES = {}
debug = 0

def register_rpn(cls, name=None):
    global REGISTERED_RPN_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_RPN_CLASSES, f"exist class: {REGISTERED_RPN_CLASSES}"
    REGISTERED_RPN_CLASSES[name] = cls
    return cls

def get_rpn_class(name):
    global REGISTERED_RPN_CLASSES
    assert name in REGISTERED_RPN_CLASSES, f"available class: {REGISTERED_RPN_CLASSES}"
    return REGISTERED_RPN_CLASSES[name]

@register_rpn
class RPN(paddle.nn.Layer):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        """deprecated. exists for checkpoint backward compilability (SECOND v1.0)
        """
        super(RPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        upsample_strides = [
            np.round(u).astype(np.int64) for u in upsample_strides
        ]
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(
                layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(
                np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])

        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        block2_input_filters = num_filters[0]
        self.block1 = Sequential(
            nn.Pad2D(1),
            nn.Conv2D(
                num_input_features, num_filters[0], 3,
                stride=layer_strides[0], bias_attr=False),
            #nn.BatchNorm2D(num_filters[0], epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        for i in range(layer_nums[0]):
            self.block1.add(
                nn.Conv2D(num_filters[0], num_filters[0], 3, padding=1, bias_attr=False))
            #self.block1.add(nn.BatchNorm2D(num_filters[0], epsilon=1e-3, momentum=0.01))
            self.block1.add(nn.ReLU())
        self.deconv1 = Sequential(
            nn.Conv2DTranspose(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0], bias_attr=False),
            #nn.BatchNorm2D(num_upsample_filters[0], epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.block2 = Sequential(
            nn.Pad2D(1),
            nn.Conv2D(
                block2_input_filters,
                num_filters[1],
                3,
                stride=layer_strides[1], bias_attr=False),
            #nn.BatchNorm2D(num_filters[1], epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.add(
                nn.Conv2D(num_filters[1], num_filters[1], 3, padding=1, bias_attr=False))
            #self.block2.add(nn.BatchNorm2D(num_filters[1], epsilon=1e-3, momentum=0.01))
            self.block2.add(nn.ReLU())
        self.deconv2 = Sequential(
            nn.Conv2DTranspose(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1], bias_attr=False),
            #nn.BatchNorm2D(num_upsample_filters[1], epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.block3 = Sequential(
            nn.Pad2D(1),
            nn.Conv2D(num_filters[1], num_filters[2], 3, stride=layer_strides[2], bias_attr=False),
            #nn.BatchNorm2D(num_filters[2], epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.add(
                nn.Conv2D(num_filters[2], num_filters[2], 3, padding=1, bias_attr=False))
            #self.block3.add(nn.BatchNorm2D(num_filters[2], epsilon=1e-3, momentum=0.01))
            self.block3.add(nn.ReLU())
        self.deconv3 = Sequential(
            nn.Conv2DTranspose(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2], bias_attr=False),
            #nn.BatchNorm2D(num_upsample_filters[2], epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2D(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2D(
            sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2D(
                sum(num_upsample_filters),
                num_anchor_per_loc * num_direction_bins, 1)

        if self._use_rc_net:
            self.conv_rc = nn.Conv2D(
                sum(num_upsample_filters), num_anchor_per_loc * box_code_size,
                1)

    def forward(self, x):
        # t = time.time()
        # torch.cuda.synchronize()

        x = self.block1(x)
        up1 = self.deconv1(x)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = paddle.concat([up1, up2, up3], axis=1)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)

        # [N, C, y(H), x(W)]
        box_preds = box_preds.transpose((0, 2, 3, 1))
        cls_preds = cls_preds.transpose((0, 2, 3, 1))
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.transpose((0, 2, 3, 1))
            ret_dict["dir_cls_preds"] = dir_cls_preds
        if self._use_rc_net:
            rc_preds = self.conv_rc(x)
            rc_preds = rc_preds.transpose((0, 2, 3, 1))
            ret_dict["rc_preds"] = rc_preds
        # torch.cuda.synchronize()
        # print("rpn forward time", time.time() - t)

        return ret_dict

flag = 1
input_count = 0
debug = 0

class RPNNoHeadBase(nn.Layer):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(RPNNoHeadBase, self).__init__()
        self._layer_strides = layer_strides
        self._num_filters = num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = upsample_strides
        self._num_upsample_filters = num_upsample_filters
        self._num_input_features = num_input_features
        self._use_norm = use_norm
        self._use_groupnorm = use_groupnorm
        self._num_groups = num_groups
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(num_upsample_filters) == len(upsample_strides)
        self._upsample_start_idx = len(layer_nums) - len(upsample_strides)
        must_equal_list = []
        for i in range(len(upsample_strides)):
            must_equal_list.append(upsample_strides[i] / np.prod(
                layer_strides[:i + self._upsample_start_idx + 1]))
        for val in must_equal_list:
            assert val == must_equal_list[0]

        in_filters = [num_input_features, *num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                num_filters[i],
                layer_num,
                stride=layer_strides[i])
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = upsample_strides[i - self._upsample_start_idx]
                if stride >= 1:
                    stride = int(np.round(stride))
                    deblock = nn.Sequential(
                        nn.Conv2DTranspose(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride, bias_attr=False),
                        nn.BatchNorm2D(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx],
                                                 epsilon=1e-3, momentum=0.01),
                        nn.ReLU(), 
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = nn.Sequential(
                        nn.Conv2D(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride, bias_attr=False),
                        nn.BatchNorm2D(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx],
                                                 epsilon=1e-3, momentum=0.01),
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self._num_out_filters = num_out_filters
        self.blocks = nn.LayerList(blocks)
        self.deblocks = nn.LayerList(deblocks)

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        raise NotImplementedError

    def forward(self, x):
        ups = []
        #stage_outputs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            #stage_outputs.append(x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))

        if len(ups) > 0:
            x = paddle.concat(ups, axis=1)
        res = {}
        for i, up in enumerate(ups):
            res[f"up{i}"] = up
        #for i, out in enumerate(stage_outputs):
        #    res[f"stage{i}"] = out
        res["out"] = x
        return res


class RPNBase(RPNNoHeadBase):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(RPNBase, self).__init__(
            use_norm=use_norm,
            num_class=num_class,
            layer_nums=layer_nums,
            layer_strides=layer_strides,
            num_filters=num_filters,
            upsample_strides=upsample_strides,
            num_upsample_filters=num_upsample_filters,
            num_input_features=num_input_features,
            num_anchor_per_loc=num_anchor_per_loc,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            box_code_size=box_code_size,
            num_direction_bins=num_direction_bins,
            name=name)
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size

        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        if len(num_upsample_filters) == 0:
            final_num_filters = self._num_out_filters
        else:
            final_num_filters = sum(num_upsample_filters)
        self.conv_cls = nn.Conv2D(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2D(final_num_filters,
                                  num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2D(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

        #for debug
        if debug:
            for i in range(len(self.blocks)):
                weight = np.load('torch_blocks' + str(i)+str(0)+'_weight.npy')
                self.blocks[i][1].weight.set_value(paddle.to_tensor(weight, stop_gradient=False))
                weight2 = np.load('torch_blocks' + str(i)+str(1)+'_weight.npy')
                self.blocks[i][2].weight.set_value(paddle.to_tensor(weight2, stop_gradient=False))
                bias = np.load('torch_blocks' + str(i)+str(1)+'_bias.npy')
                self.blocks[i][2].bias.set_value(paddle.to_tensor(bias, stop_gradient=False))
                for j in range(int((len(self.blocks[i]) - 4) / 3)):
                    weight = np.load('torch_blocks' + str(i)+str(2+2*j+0)+'_weight.npy')
                    weight2 = np.load('torch_blocks' + str(i)+str(2+2*j+1)+'_weight.npy')
                    bias = np.load('torch_blocks' + str(i)+str(2+2*j+1)+'_bias.npy')
                    self.blocks[i][4 + j*3].weight.set_value(paddle.to_tensor(weight, stop_gradient=False))
                    self.blocks[i][4 + j*3+1].weight.set_value(paddle.to_tensor(weight2, stop_gradient=False))
                    self.blocks[i][4 + j*3+1].bias.set_value(paddle.to_tensor(bias, stop_gradient=False))
            for i in range(len(self.deblocks)):
                for j in range(int(len(self.deblocks[i])/3)):
                    weight = np.load('torch_deblocks' + str(i)+str(2*j)+'_weight.npy')
                    weight2 = np.load('torch_deblocks' + str(i)+str(2*j+1)+'_weight.npy')
                    bias = np.load('torch_deblocks' + str(i)+str(2*j+1)+'_bias.npy')
                    self.deblocks[i][j*3].weight.set_value(paddle.to_tensor(weight, stop_gradient=False))
                    self.deblocks[i][j*3+1].weight.set_value(paddle.to_tensor(weight2, stop_gradient=False))
                    self.deblocks[i][j*3+1].bias.set_value(paddle.to_tensor(bias, stop_gradient=False))

            conv_box_weight = np.load("torch_conv_box_weight.npy")
            conv_box_bias = np.load("torch_conv_box_bias.npy")
            conv_cls_weight = np.load("torch_conv_cls_weight.npy")
            conv_cls_bias = np.load("torch_conv_cls_bias.npy")
            self.conv_box.weight.set_value(paddle.to_tensor(conv_box_weight, stop_gradient=False))
            self.conv_box.bias.set_value(paddle.to_tensor(conv_box_bias, stop_gradient=False))
            self.conv_cls.weight.set_value(paddle.to_tensor(conv_cls_weight, stop_gradient=False))
            self.conv_cls.bias.set_value(paddle.to_tensor(conv_cls_bias, stop_gradient=False))
            if self._use_direction_classifier:
                conv_dir_cls_weight = np.load("torch_conv_dir_cls_weight.npy")
                conv_dir_cls_bias = np.load("torch_conv_dir_cls_bias.npy")
                self.conv_dir_cls.weight.set_value(paddle.to_tensor(conv_dir_cls_weight, stop_gradient=False))
                self.conv_dir_cls.bias.set_value(paddle.to_tensor(conv_dir_cls_bias, stop_gradient=False))

    def forward(self, x):
        global flag
        global input_count
        #for debug
        if debug:
            base_dir = './rpn/' + str(input_count) + '_'
            for i in range(len(self.blocks)):
                weight = np.load(base_dir + 'torch_blocks' + str(i)+str(0)+'_weight.npy')
                #self.blocks[i][1].weight.set_value(paddle.to_tensor(weight, stop_gradient=False))
                assert np.allclose(weight,
                self.blocks[i][1].weight.numpy(), atol=1e-3,
                rtol=1e-3)
                weight2 = np.load(base_dir + 'torch_blocks' + str(i)+str(1)+'_weight.npy')
                #self.blocks[i][2].weight.set_value(paddle.to_tensor(weight2, stop_gradient=False))
                assert np.allclose(weight2, self.blocks[i][2].weight.numpy(), atol=1e-3, rtol=1e-3)
                bias = np.load(base_dir + 'torch_blocks' + str(i)+str(1)+'_bias.npy')
                #self.blocks[i][2].bias.set_value(paddle.to_tensor(bias, stop_gradient=False))
                assert np.allclose(bias, self.blocks[i][2].bias.numpy(), atol=1e-3, rtol=1e-3)
                for j in range(int((len(self.blocks[i]) - 4) / 3)):
                    weight = np.load(base_dir + 'torch_blocks' + str(i)+str(2+2*j+0)+'_weight.npy')
                    weight2 = np.load(base_dir + 'torch_blocks' + str(i)+str(2+2*j+1)+'_weight.npy')
                    bias = np.load(base_dir + 'torch_blocks' + str(i)+str(2+2*j+1)+'_bias.npy')
                    #self.blocks[i][4 + j*3].weight.set_value(paddle.to_tensor(weight, stop_gradient=False))
                    assert np.allclose(weight,
                    self.blocks[i][4+j*3].weight.numpy(), atol=1e-3,
                    rtol=1e-3)
                    #self.blocks[i][4 + j*3+1].weight.set_value(paddle.to_tensor(weight2, stop_gradient=False))
                    assert np.allclose(weight2, self.blocks[i][4+j*3+1].weight.numpy(), atol=1e-3, rtol=1e-3)
                    #self.blocks[i][4 + j*3+1].bias.set_value(paddle.to_tensor(bias, stop_gradient=False))
                    assert np.allclose(bias, self.blocks[i][4+j*3+1].bias.numpy(), atol=1e-3, rtol=1e-3)
            for i in range(len(self.deblocks)):
                for j in range(int(len(self.deblocks[i])/3)):
                    weight = np.load(base_dir + 'torch_deblocks' + str(i)+str(2*j)+'_weight.npy')
                    weight2 = np.load(base_dir + 'torch_deblocks' + str(i)+str(2*j+1)+'_weight.npy')
                    bias = np.load(base_dir + 'torch_deblocks' + str(i)+str(2*j+1)+'_bias.npy')
                    #self.deblocks[i][j*3].weight.set_value(paddle.to_tensor(weight, stop_gradient=False))
                    assert np.allclose(weight,
                    self.deblocks[i][j*3].weight.numpy(), atol=1e-3,
                    rtol=1e-3)
                    #self.deblocks[i][j*3+1].weight.set_value(paddle.to_tensor(weight2, stop_gradient=False))
                    assert np.allclose(weight2, self.deblocks[i][j*3+1].weight.numpy(), atol=1e-3, rtol=1e-3)
                    #self.deblocks[i][j*3+1].bias.set_value(paddle.to_tensor(bias, stop_gradient=False))
                    assert np.allclose(bias, self.deblocks[i][j*3+1].bias.numpy(), atol=1e-3, rtol=1e-3)

            conv_box_weight = np.load(base_dir + "torch_conv_box_weight.npy")
            conv_box_bias = np.load(base_dir + "torch_conv_box_bias.npy")
            conv_cls_weight = np.load(base_dir + "torch_conv_cls_weight.npy")
            conv_cls_bias = np.load(base_dir + "torch_conv_cls_bias.npy")
            self.conv_box.weight.set_value(paddle.to_tensor(conv_box_weight, stop_gradient=False))
            self.conv_box.bias.set_value(paddle.to_tensor(conv_box_bias, stop_gradient=False))
            self.conv_cls.weight.set_value(paddle.to_tensor(conv_cls_weight, stop_gradient=False))
            self.conv_cls.bias.set_value(paddle.to_tensor(conv_cls_bias, stop_gradient=False))
            if self._use_direction_classifier:
                conv_dir_cls_weight = np.load(base_dir + "torch_conv_dir_cls_weight.npy")
                conv_dir_cls_bias = np.load(base_dir + "torch_conv_dir_cls_bias.npy")
                self.conv_dir_cls.weight.set_value(paddle.to_tensor(conv_dir_cls_weight, stop_gradient=False))
                self.conv_dir_cls.bias.set_value(paddle.to_tensor(conv_dir_cls_bias, stop_gradient=False))

        res = super().forward(x)
        x = res["out"]

        #if debug:
            #torch_rpn_out = np.load('./rpn/' + str(input_count) + "_out.npy")
            #assert np.allclose(torch_rpn_out, x.numpy(), atol=1e-1, rtol=1e-1)
            #input_count += 1
            #print("compare rpn_out success")
        

        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.reshape((-1, self._num_anchor_per_loc,
                                   self._box_code_size, H, W)).transpose((
                                       0, 1, 3, 4, 2))
        cls_preds = cls_preds.reshape((-1, self._num_anchor_per_loc,
                                   self._num_class, H, W)).transpose((
                                       0, 1, 3, 4, 2))
        # box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        # cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()

        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }

        if flag == 0:
            torch_box_preds = np.load("torch_box_preds.npy")
            torch_cls_preds = np.load("torch_cls_preds.npy")
            assert np.allclose(torch_box_preds, box_preds.numpy(), atol=1e-5, rtol=1e-5)
            assert np.allclose(torch_cls_preds, cls_preds.numpy(), atol=1e-5, rtol=1e-5)
            print("compare box_preds and cls_preds success")
            flag = 1

        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.reshape((
                -1, self._num_anchor_per_loc, self._num_direction_bins, H,
                W)).transpose((0, 1, 3, 4, 2))
            # dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        return ret_dict


@register_rpn
class RPNV2(RPNBase):
    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        block = Sequential(
            #nn.ZeroPad2d(1),
            nn.Pad2D(1),
            nn.Conv2D(inplanes, planes, 3, stride=stride, bias_attr=False),
            nn.BatchNorm2D(planes, epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False))
            block.add(nn.BatchNorm2D(planes, epsilon=1e-3, momentum=0.01))
            block.add(nn.ReLU())

        return block, planes


def test():
    spatial_features = np.load("torch_spatial_features.npy")
    spatial_features = paddle.to_tensor(spatial_features, stop_gradient=False)
    rpn = RPNV2() 
    out = rpn(spatial_features)
    out['cls_preds'].sum().backward()
    print(spatial_features.grad.shape)

    
#test()
