import numpy as np
import contextlib
import paddle
import time
from paddle import nn
import paddle.nn.functional as F
from enum import Enum
import torchplus
from second.pytorch.core import box_paddle_ops

from second.pytorch.core.losses import (WeightedSigmoidClassificationLoss,
                                        WeightedSmoothL1LocalizationLoss,
                                        WeightedSoftmaxClassificationLoss)
from second.pytorch.models import middle, pointpillars, rpn, voxel_encoder
from torchplus import metrics
from second.pytorch.utils import paddle_timer

loss_flag=1
input_count = 0
debug = 0
profiling=False

def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = paddle.cast((labels > 0), cls_loss.dtype) * cls_loss.reshape((
            batch_size, -1))
        cls_neg_loss = paddle.cast((labels == 0), cls_loss.dtype) * cls_loss.reshape((
            batch_size, -1))
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss

REGISTERED_NETWORK_CLASSES = {}

def register_voxelnet(cls, name=None):
    global REGISTERED_NETWORK_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_NETWORK_CLASSES, f"exist class: {REGISTERED_NETWORK_CLASSES}"
    REGISTERED_NETWORK_CLASSES[name] = cls
    return cls

def get_voxelnet_class(name):
    global REGISTERED_NETWORK_CLASSES
    assert name in REGISTERED_NETWORK_CLASSES, f"available class: {REGISTERED_NETWORK_CLASSES}"
    return REGISTERED_NETWORK_CLASSES[name]

class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"
    DontNorm = "dont_norm"

flag = 1

@register_voxelnet
class VoxelNet(nn.Layer):
    def __init__(self,
                 output_shape,
                 num_class=2,
                 num_input_features=4,
                 vfe_class_name="VoxelFeatureExtractor",
                 vfe_num_filters=[32, 128],
                 with_distance=False,
                 middle_class_name="SparseMiddleExtractor",
                 middle_num_input_features=-1,
                 middle_num_filters_d1=[64],
                 middle_num_filters_d2=[64, 64],
                 rpn_class_name="RPN",
                 rpn_num_input_features=-1,
                 rpn_layer_nums=[3, 5, 5],
                 rpn_layer_strides=[2, 2, 2],
                 rpn_num_filters=[128, 128, 256],
                 rpn_upsample_strides=[1, 2, 4],
                 rpn_num_upsample_filters=[256, 256, 256],
                 use_norm=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_direction_classifier=True,
                 use_sigmoid_score=False,
                 encode_background_as_zeros=True,
                 use_rotate_nms=True,
                 multiclass_nms=False,
                 nms_score_thresholds=None,
                 nms_pre_max_sizes=None,
                 nms_post_max_sizes=None,
                 nms_iou_thresholds=None,
                 target_assigner=None,
                 cls_loss_weight=1.0,
                 loc_loss_weight=1.0,
                 pos_cls_weight=1.0,
                 neg_cls_weight=1.0,
                 direction_loss_weight=1.0,
                 loss_norm_type=LossNormType.NormByNumPositives,
                 encode_rad_error_by_sin=False,
                 loc_loss_ftor=None,
                 cls_loss_ftor=None,
                 measure_time=False,
                 voxel_generator=None,
                 post_center_range=None,
                 dir_offset=0.0,
                 sin_error_factor=1.0,
                 nms_class_agnostic=False,
                 num_direction_bins=2,
                 direction_limit_offset=0,
                 name='voxelnet'):
        super().__init__()
        self.name = name
        self._sin_error_factor = sin_error_factor
        self._num_class = num_class
        self._use_rotate_nms = use_rotate_nms
        self._multiclass_nms = multiclass_nms
        self._nms_score_thresholds = nms_score_thresholds
        self._nms_pre_max_sizes = nms_pre_max_sizes
        self._nms_post_max_sizes = nms_post_max_sizes
        self._nms_iou_thresholds = nms_iou_thresholds
        self._use_sigmoid_score = use_sigmoid_score
        self._encode_background_as_zeros = encode_background_as_zeros
        self._use_direction_classifier = use_direction_classifier
        self._num_input_features = num_input_features
        self._box_coder = target_assigner.box_coder
        self.target_assigner = target_assigner
        self.voxel_generator = voxel_generator
        self._pos_cls_weight = pos_cls_weight
        self._neg_cls_weight = neg_cls_weight
        self._encode_rad_error_by_sin = encode_rad_error_by_sin
        self._loss_norm_type = loss_norm_type
        self._dir_loss_ftor = WeightedSoftmaxClassificationLoss()
        self._diff_loc_loss_ftor = WeightedSmoothL1LocalizationLoss()
        self._dir_offset = dir_offset
        self._loc_loss_ftor = loc_loss_ftor
        self._cls_loss_ftor = cls_loss_ftor
        self._direction_loss_weight = direction_loss_weight
        self._cls_loss_weight = cls_loss_weight
        self._loc_loss_weight = loc_loss_weight
        self._post_center_range = post_center_range or []
        self.measure_time = measure_time
        self._nms_class_agnostic = nms_class_agnostic
        self._num_direction_bins = num_direction_bins
        self._dir_limit_offset = direction_limit_offset
        self.voxel_feature_extractor = voxel_encoder.get_vfe_class(vfe_class_name)(
            num_input_features,
            use_norm,
            num_filters=vfe_num_filters,
            with_distance=with_distance,
            voxel_size=self.voxel_generator.voxel_size,
            pc_range=self.voxel_generator.point_cloud_range,
        )
        print("vfe:")
        print(self.voxel_feature_extractor)
        print(len(self.voxel_feature_extractor.parameters()))
        self.middle_feature_extractor = middle.get_middle_class(middle_class_name)(
            output_shape,
            use_norm,
            num_input_features=middle_num_input_features,
            num_filters_down1=middle_num_filters_d1,
            num_filters_down2=middle_num_filters_d2)
        print("middle:")
        print(self.middle_feature_extractor)
        print(len(self.middle_feature_extractor.parameters()))
        self.rpn = rpn.get_rpn_class(rpn_class_name)(
            use_norm=True,
            num_class=num_class,
            layer_nums=rpn_layer_nums,
            layer_strides=rpn_layer_strides,
            num_filters=rpn_num_filters,
            upsample_strides=rpn_upsample_strides,
            num_upsample_filters=rpn_num_upsample_filters,
            num_input_features=rpn_num_input_features,
            num_anchor_per_loc=target_assigner.num_anchors_per_location,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            box_code_size=target_assigner.box_coder.code_size,
            num_direction_bins=self._num_direction_bins)
        print("rpn:")
        print(self.rpn)
        print(len(self.rpn.parameters()))
        self.rpn_acc = metrics.Accuracy(
            dim=-1, encode_background_as_zeros=encode_background_as_zeros)
        self.rpn_precision = metrics.Precision(dim=-1)
        self.rpn_recall = metrics.Recall(dim=-1)
        self.rpn_metrics = metrics.PrecisionRecall(
            dim=-1,
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            use_sigmoid_score=use_sigmoid_score,
            encode_background_as_zeros=encode_background_as_zeros)

        self.rpn_cls_loss = metrics.Scalar()
        self.rpn_loc_loss = metrics.Scalar()
        self.rpn_total_loss = metrics.Scalar()
        #self.register_buffer("global_step", torch.LongTensor(1).zero_())
        self.register_buffer("global_step", paddle.zeros(shape=(1,), dtype='int64'))


        self._time_dict = {}
        self._time_total_dict = {}
        self._time_count_dict = {}

    def start_timer(self, *names):
        if not self.measure_time:
            return
        #torch.cuda.synchronize()
        paddle.device.cuda.synchronize()
        for name in names:
            self._time_dict[name] = time.time()

    def end_timer(self, name):
        if not self.measure_time:
            return
        #torch.cuda.synchronize()
        paddle.device.cuda.synchronize()
        time_elapsed = time.time() - self._time_dict[name]
        if name not in self._time_count_dict:
            self._time_count_dict[name] = 1
            self._time_total_dict[name] = time_elapsed
        else:
            self._time_count_dict[name] += 1
            self._time_total_dict[name] += time_elapsed
        self._time_dict[name] = 0

    def clear_timer(self):
        self._time_count_dict.clear()
        self._time_dict.clear()
        self._time_total_dict.clear()

    @contextlib.contextmanager
    def profiler(self):
        old_measure_time = self.measure_time
        self.measure_time = True
        yield
        self.measure_time = old_measure_time

    def get_avg_time_dict(self):
        ret = {}
        for name, val in self._time_total_dict.items():
            count = self._time_count_dict[name]
            ret[name] = val / max(1, count)
        return ret

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def clear_global_step(self):
        self.global_step.zero_()

    def loss(self, example, preds_dict):
        box_preds = preds_dict["box_preds"]
        cls_preds = preds_dict["cls_preds"]
        batch_size_dev = cls_preds.shape[0]
        self.start_timer("loss forward")
        labels = example['labels']
        reg_targets = example['reg_targets']
        importance = example['importance']
        self.start_timer("prepare weight forward")

        global loss_flag
        if loss_flag == 0:
            torch_labels = np.load("torch_labels.npy")
            if np.allclose(torch_labels, labels.numpy()) is False:
                print("compare labels failed...")
                labels = paddle.to_tensor(torch_labels)
            torch_reg_targets = np.load("torch_reg_targets.npy")
            if np.allclose(torch_reg_targets, reg_targets.numpy()) is False:
                print("compare reg_targets failed...")
                reg_targets = paddle.to_tensor(torch_reg_targets)
            torch_importance = np.load("torch_importance.npy")
            if np.allclose(torch_importance, importance.numpy()) is False:
                print("compare importance failed...")
                importance = paddle.to_tensor(torch_importance)

        cls_weights, reg_weights, cared = prepare_loss_weights(
            labels,
            pos_cls_weight=self._pos_cls_weight,
            neg_cls_weight=self._neg_cls_weight,
            loss_norm_type=self._loss_norm_type,
            dtype=box_preds.dtype)

        if loss_flag == 0:
            torch_cls_weights = np.load("torch_cls_weights.npy")
            assert np.allclose(torch_cls_weights, cls_weights.numpy(), atol=1e-5, rtol=1e-5)
            torch_reg_weights = np.load("torch_reg_weights.npy")
            assert np.allclose(torch_reg_weights, reg_weights.numpy(), atol=1e-5, rtol=1e-5)
            torch_cared_weight = np.load("torch_cared_weights.npy")
            assert np.allclose(torch_cared_weight, cared.numpy(), atol=1e-5, rtol=1e-5)
            print("compared loss weight success")

        cls_targets = paddle.cast(labels * cared, labels.dtype)
        cls_targets = cls_targets.unsqueeze(-1)
        self.end_timer("prepare weight forward")
        self.start_timer("create_loss forward")
        loc_loss, cls_loss = create_loss(
            self._loc_loss_ftor,
            self._cls_loss_ftor,
            box_preds=box_preds,
            cls_preds=cls_preds,
            cls_targets=cls_targets,
            cls_weights=cls_weights * importance,
            reg_targets=reg_targets,
            reg_weights=reg_weights * importance,
            num_class=self._num_class,
            encode_rad_error_by_sin=self._encode_rad_error_by_sin,
            encode_background_as_zeros=self._encode_background_as_zeros,
            box_code_size=self._box_coder.code_size,
            sin_error_factor=self._sin_error_factor,
            num_direction_bins=self._num_direction_bins,
        )
        loc_loss_reduced = loc_loss.sum() / batch_size_dev
        loc_loss_reduced *= self._loc_loss_weight
        cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
        cls_pos_loss /= self._pos_cls_weight
        cls_neg_loss /= self._neg_cls_weight
        cls_loss_reduced = cls_loss.sum() / batch_size_dev
        cls_loss_reduced *= self._cls_loss_weight
        loss = loc_loss_reduced + cls_loss_reduced
        self.end_timer("create_loss forward")
        if self._use_direction_classifier:
            dir_targets = get_direction_target(
                example['anchors'],
                reg_targets,
                dir_offset=self._dir_offset,
                num_bins=self._num_direction_bins)
            dir_logits = preds_dict["dir_cls_preds"].reshape((
                batch_size_dev, -1, self._num_direction_bins))
            weights = paddle.cast((labels > 0), dir_logits.dtype) * importance
            weights /= paddle.clip(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self._dir_loss_ftor(
                dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size_dev
            loss += dir_loss * self._direction_loss_weight
        self.end_timer("loss forward")
        res = {
            "loss": loss,
            "cls_loss": cls_loss,
            "loc_loss": loc_loss,
            "cls_pos_loss": cls_pos_loss,
            "cls_neg_loss": cls_neg_loss,
            "cls_preds": cls_preds,
            "cls_loss_reduced": cls_loss_reduced,
            "loc_loss_reduced": loc_loss_reduced,
            "cared": cared,
        }
        if self._use_direction_classifier:
            res["dir_loss_reduced"] = dir_loss

        if loss_flag == 0:
            loss_flag = 1
            torch_cls_loss = np.load("torch_cls_loss.npy")
            assert np.allclose(torch_cls_loss, cls_loss.numpy(), atol=1e-5, rtol=1e-5)
            torch_loc_loss = np.load("torch_loc_loss.npy")
            assert np.allclose(torch_loc_loss, loc_loss.numpy(), atol=1e-2, rtol=1e-2)
            torch_cls_pos_loss = np.load("torch_cls_pos_loss.npy")
            assert np.allclose(torch_cls_pos_loss, cls_pos_loss.numpy(), atol=1e-2, rtol=1e-2)
            torch_cls_neg_loss = np.load("torch_cls_neg_loss.npy")
            assert np.allclose(torch_cls_neg_loss, cls_neg_loss.numpy(), atol=1e-2, rtol=1e-2)
            torch_loss = np.load("torch_loss.npy")
            assert np.allclose(torch_loss, loss.numpy(), atol=1e-2, rtol=1e-2)
            torch_cls_loss_reduced = np.load("torch_cls_loss_reduced.npy")
            assert np.allclose(torch_cls_loss_reduced, cls_loss_reduced.numpy(), atol=1e-2, rtol=1e-2)
            torch_loc_loss_reduced = np.load("torch_loc_loss_reduced.npy")
            assert np.allclose(torch_loc_loss_reduced, loc_loss_reduced.numpy(), atol=1e-5, rtol=1e-5)
            print("compare loss success")
        return res

    def network_forward(self, voxels, num_points, coors, batch_size):
        """this function is used for subclass.
        you can add custom network architecture by subclass VoxelNet class
        and override this function.
        Returns: 
            preds_dict: {
                box_preds: ...
                cls_preds: ...
                dir_cls_preds: ...
            }
        """
        self.start_timer("voxel_feature_extractor")
        global flag
        global input_count
        voxels.stop_gradient=False

        if debug:
            torch_voxels = np.load('./voxel/' + str(input_count) + '_voxels.npy')
            assert np.allclose(torch_voxels, voxels.numpy(),
            atol=1e-5, rtol=1e-5)
            print("compare voxels " + str(input_count) + " success")

        if profiling:
            t0 = time.time()
        voxel_features = self.voxel_feature_extractor(voxels, num_points,
                                                      coors)

        if profiling:
            paddle.device.cuda.synchronize()
            t1 = time.time()
            print("vfe time:", t1-t0)

        if debug:
            torch_voxel_features = np.load('./vfe/' + str(input_count) + '_voxel_features.npy')
            assert np.allclose(torch_voxel_features,
            voxel_features.numpy(), atol=1e-5, rtol=1e-5)
            print("compared voxel_features " + str(input_count) + " success")

        self.end_timer("voxel_feature_extractor")

        self.start_timer("middle forward")

        if profiling:
            t0 = time.time()

        self.spatial_features = self.middle_feature_extractor(
            voxel_features, coors, batch_size)

        if profiling:
            paddle.device.cuda.synchronize()
            t1 = time.time()
            print("middle time:", t1-t0)

        if debug:
            torch_spatial_features = np.load('./middle/' + str(input_count) + '_spatial_features.npy')
            assert np.allclose(torch_spatial_features, self.spatial_features.numpy(), atol=1e-5,
            rtol=1e-5)
            print("compared spatial_features " + str(input_count) + " success")


        self.end_timer("middle forward")

        self.start_timer("rpn forward")
        if profiling:
            t0 = time.time()
        preds_dict = self.rpn(self.spatial_features)
        if profiling:
            paddle.device.cuda.synchronize()
            t1 = time.time()
            print("rpn time:", t1-t0)

        self.end_timer("rpn forward")
        #print("voxelnet forward out.shape=", preds_dict.shape)

        if debug:
            #torch_box_preds = np.load('./rpn/' + str(input_count) + '_box_preds.npy')
            #assert np.allclose(torch_box_preds, preds_dict['box_preds'].numpy(), atol=1e-1, rtol=1e-1)
            #print("compared box_preds " + str(input_count) + " success")
            torch_cls_preds = np.load('./rpn/' + str(input_count) + '_cls_preds.npy')
            assert np.allclose(torch_cls_preds,
            preds_dict['cls_preds'].numpy(), atol=1e-3,
            rtol=1e-3)
            print("compared cls_preds " + str(input_count) + " success")
            input_count += 1
        return preds_dict

    def forward(self, example):
        """module's forward should always accept dict and return loss.
        """
        voxels = example["voxels"]
        num_points = example["num_points"]
        coors = example["coordinates"]
        if len(num_points.shape) == 2:  # multi-gpu
            print("len of num_points.shape = ", len(num_points.shape))
            num_voxel_per_batch = example["num_voxels"].cpu().numpy().reshape(
                -1)
            voxel_list = []
            num_points_list = []
            coors_list = []
            for i, num_voxel in enumerate(num_voxel_per_batch):
                voxel_list.append(voxels[i, :num_voxel])
                num_points_list.append(num_points[i, :num_voxel])
                coors_list.append(coors[i, :num_voxel])
            voxels = paddle.concat(voxel_list, axis=0)
            num_points = paddle.concat(num_points_list, axis=0)
            coors = paddle.concat(coors_list, axis=0)
        batch_anchors = example["anchors"]
        batch_size_dev = batch_anchors.shape[0]
        # features: [num_voxels, max_num_points_per_voxel, 7]
        # num_points: [num_voxels]
        # coors: [num_voxels, 4]
        preds_dict = self.network_forward(voxels, num_points, coors, batch_size_dev)
        # need to check size.
        box_preds = preds_dict["box_preds"].reshape((batch_size_dev, -1, self._box_coder.code_size))
        err_msg = f"num_anchors={batch_anchors.shape[1]}, but num_output={box_preds.shape[1]}. please check size"
        assert batch_anchors.shape[1] == box_preds.shape[1], err_msg
        if self.training:
            #print("is training...\n")
            return self.loss(example, preds_dict)
        else:
            self.start_timer("predict")
            with paddle.no_grad():
                res = self.predict(example, preds_dict)
            self.end_timer("predict")
            return res

    def predict(self, example, preds_dict):
        """start with v1.6.0, this function don't contain any kitti-specific code.
        Returns:
            predict: list of pred_dict.
            pred_dict: {
                box3d_lidar: [N, 7] 3d box.
                scores: [N]
                label_preds: [N]
                metadata: meta-data which contains dataset-specific information.
                    for kitti, it contains image idx (label idx), 
                    for nuscenes, sample_token is saved in it.
            }
        """
        batch_size = example['anchors'].shape[0]
        if "metadata" not in example or len(example["metadata"]) == 0:
            meta_list = [None] * batch_size
        else:
            meta_list = example["metadata"]
        batch_anchors = example["anchors"].reshape((batch_size, -1,
                                                example["anchors"].shape[-1]))
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].reshape((batch_size, -1))

        t = time.time()
        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.reshape((batch_size, -1,
                                               self._box_coder.code_size))
        num_class_with_bg = self._num_class
        if not self._encode_background_as_zeros:
            num_class_with_bg = self._num_class + 1

        batch_cls_preds = batch_cls_preds.reshape((batch_size, -1,
                                               num_class_with_bg))
        batch_box_preds = self._box_coder.decode_paddle(batch_box_preds,
                                                       batch_anchors)
        if self._use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.reshape((batch_size, -1,
                                                   self._num_direction_bins))
        else:
            batch_dir_preds = [None] * batch_size

        predictions_dicts = []
        post_center_range = None
        if len(self._post_center_range) > 0:
            post_center_range = paddle.to_tensor(
                self._post_center_range,
                dtype=batch_box_preds.dtype)
        for box_preds, cls_preds, dir_preds, a_mask, meta in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds,
                batch_anchors_mask, meta_list):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            #box_preds = box_preds.float()
            box_preds = paddle.cast(box_preds, dtype='float32')
            #cls_preds = cls_preds.float()
            cls_preds = paddle.cast(cls_preds, dtype='float32')
            if self._use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                dir_labels = paddle.argmax(dir_preds, axis=-1)
            if self._encode_background_as_zeros:
                # this don't support softmax
                assert self._use_sigmoid_score is True
                total_scores = F.sigmoid(cls_preds)
            else:
                # encode background as first element in one-hot vector
                if self._use_sigmoid_score:
                    total_scores = F.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
            # Apply NMS in birdeye view
            if self._use_rotate_nms:
                nms_func = box_paddle_ops.rotate_nms
            else:
                nms_func = box_paddle_ops.nms
            feature_map_size_prod = batch_box_preds.shape[
                1] // self.target_assigner.num_anchors_per_location
            if self._multiclass_nms:
                assert self._encode_background_as_zeros is True
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                if not self._use_rotate_nms:
                    box_preds_corners = box_paddle_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4])
                    boxes_for_nms = box_paddle_ops.corner_to_standup_nd(
                        box_preds_corners)

                selected_boxes, selected_labels, selected_scores = [], [], []
                selected_dir_labels = []

                scores = total_scores
                boxes = boxes_for_nms
                selected_per_class = []
                score_threshs = self._nms_score_thresholds
                pre_max_sizes = self._nms_pre_max_sizes
                post_max_sizes = self._nms_post_max_sizes
                iou_thresholds = self._nms_iou_thresholds
                for class_idx, score_thresh, pre_ms, post_ms, iou_th in zip(
                        range(self._num_class),
                        score_threshs,
                        pre_max_sizes, post_max_sizes, iou_thresholds):
                    if self._nms_class_agnostic:
                        class_scores = total_scores.reshape((
                            feature_map_size_prod, -1,
                            self._num_class))[..., class_idx]
                        class_scores = class_scores.reshape((-1))
                        class_boxes_nms = boxes.reshape((-1,
                                                     boxes_for_nms.shape[-1]))
                        class_boxes = box_preds
                        class_dir_labels = dir_labels
                    else:
                        anchors_range = self.target_assigner.anchors_range(class_idx)
                        class_scores = total_scores.reshape((
                            -1,
                            self._num_class))[anchors_range[0]:anchors_range[1], class_idx]
                        class_boxes_nms = boxes.reshape((-1,
                            boxes_for_nms.shape[-1]))[anchors_range[0]:anchors_range[1], :]
                        class_scores = class_scores.contiguous().reshape((-1))
                        class_boxes_nms = class_boxes_nms.reshape((
                            -1, boxes_for_nms.shape[-1]))
                        class_boxes = box_preds.reshape((-1,
                            box_preds.shape[-1]))[anchors_range[0]:anchors_range[1], :]
                        class_boxes = class_boxes.contiguous().reshape((
                            -1, box_preds.shape[-1]))
                        if self._use_direction_classifier:
                            class_dir_labels = dir_labels.reshape((-1))[anchors_range[0]:anchors_range[1]]
                            class_dir_labels = class_dir_labels.contiguous(
                            ).reshape((-1))
                    if score_thresh > 0.0:
                        class_scores_keep = class_scores >= score_thresh
                        if class_scores_keep.shape[0] == 0:
                            selected_per_class.append(None)
                            continue
                        class_scores = class_scores[class_scores_keep]
                    if class_scores.shape[0] != 0:
                        if score_thresh > 0.0:
                            class_boxes_nms = class_boxes_nms[
                                class_scores_keep]
                            class_boxes = class_boxes[class_scores_keep]
                            class_dir_labels = class_dir_labels[
                                class_scores_keep]
                        keep = nms_func(class_boxes_nms, class_scores, pre_ms,
                                        post_ms, iou_th)
                        if keep.shape[0] != 0:
                            selected_per_class.append(keep)
                        else:
                            selected_per_class.append(None)
                    else:
                        selected_per_class.append(None)
                    selected = selected_per_class[-1]

                    if selected is not None:
                        selected_boxes.append(class_boxes[selected])
                        selected_labels.append(
                            paddle.full([class_boxes[selected].shape[0]], class_idx,
                            dtype='int64'))
                        if self._use_direction_classifier:
                            selected_dir_labels.append(
                                class_dir_labels[selected])
                        selected_scores.append(class_scores[selected])
                selected_boxes = paddle.concat(selected_boxes, axis=0)
                selected_labels = paddle.concat(selected_labels, axis=0)
                selected_scores = paddle.concat(selected_scores, axis=0)
                if self._use_direction_classifier:
                    selected_dir_labels = paddle.concat(selected_dir_labels, axis=0)
            else:
                # get highest score per prediction, than apply nms
                # to remove overlapped box.
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(-1)
                    shape = [total_scores.shape[0]]
                    top_labels = paddle.zeros(shape, dtype='int64')
                else:
                    top_scores, top_labels = paddle.max(
                        total_scores, axis=-1)
                if self._nms_score_thresholds[0] > 0.0:
                    top_scores_keep = top_scores >= self._nms_score_thresholds[0]
                    top_scores = top_scores.masked_select(top_scores_keep)

                if top_scores.shape[0] != 0:
                    if self._nms_score_thresholds[0] > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        if self._use_direction_classifier:
                            dir_labels = dir_labels[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    #boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                    boxes_for_nms = paddle.index_select(box_preds, axis=len(box_preds.shape)-1, index=paddle.to_tensor([0, 1, 3, 4, 6]))
                    if not self._use_rotate_nms:
                        box_preds_corners = box_paddle_ops.center_to_corner_box2d(
                            boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                            boxes_for_nms[:, 4])
                        boxes_for_nms = box_paddle_ops.corner_to_standup_nd(
                            box_preds_corners)
                    # the nms in 3d detection just remove overlap boxes.
                    selected = nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=self._nms_pre_max_sizes[0],
                        post_max_size=self._nms_post_max_sizes[0],
                        iou_threshold=self._nms_iou_thresholds[0],
                    )
                else:
                    selected = []
                # if selected is not None:
                selected_boxes = box_preds[selected]
                if self._use_direction_classifier:
                    selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]
            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self._use_direction_classifier:
                    dir_labels = selected_dir_labels
                    period = (2 * np.pi / self._num_direction_bins)
                    dir_rot = box_paddle_ops.limit_period(
                        box_preds[..., 6] - self._dir_offset,
                        self._dir_limit_offset, period)
                    tmp = dir_rot + self._dir_offset + period * dir_labels#.to(box_preds.dtype)
                    box_preds[
                        ...,
                        6] = paddle.cast(tmp, dtype=box_preds.dtype)
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = {
                        "box3d_lidar": final_box_preds[mask],
                        "scores": final_scores[mask],
                        "label_preds": label_preds[mask],
                        "metadata": meta,
                    }
                else:
                    predictions_dict = {
                        "box3d_lidar": final_box_preds,
                        "scores": final_scores,
                        "label_preds": label_preds,
                        "metadata": meta,
                    }
            else:
                dtype = batch_box_preds.dtype
                device = batch_box_preds.device
                predictions_dict = {
                    "box3d_lidar":
                    paddle.zeros([0, box_preds.shape[-1]],
                                dtype=dtype,
                                device=device),
                    "scores":
                    paddle.zeros([0], dtype=dtype, device=device),
                    "label_preds":
                    paddle.zeros([0], dtype=top_labels.dtype),
                    "metadata":
                    meta,
                }
            predictions_dicts.append(predictions_dict)
        return predictions_dicts

    def metrics_to_float(self):
        self.rpn_acc.float()
        self.rpn_metrics.float()
        self.rpn_cls_loss.float()
        self.rpn_loc_loss.float()
        self.rpn_total_loss.float()

    def update_metrics(self, cls_loss, loc_loss, cls_preds, labels, sampled):
        batch_size = cls_preds.shape[0]
        num_class = self._num_class
        if not self._encode_background_as_zeros:
            num_class += 1
        cls_preds = cls_preds.reshape((batch_size, -1, num_class))
        rpn_acc = self.rpn_acc(labels, cls_preds, sampled).numpy()[0]
        prec, recall = self.rpn_metrics(labels, cls_preds, sampled)
        prec = prec.numpy()
        recall = recall.numpy()
        rpn_cls_loss = self.rpn_cls_loss(cls_loss).numpy()[0]
        rpn_loc_loss = self.rpn_loc_loss(loc_loss).numpy()[0]
        ret = {
            "loss": {
                "cls_loss": float(rpn_cls_loss),
                "cls_loss_rt": float(cls_loss.numpy()),
                'loc_loss': float(rpn_loc_loss),
                "loc_loss_rt": float(loc_loss.numpy()),
            },
            "rpn_acc": float(rpn_acc),
            "pr": {},
        }
        for i, thresh in enumerate(self.rpn_metrics.thresholds):
            ret["pr"][f"prec@{int(thresh*100)}"] = float(prec[i])
            ret["pr"][f"rec@{int(thresh*100)}"] = float(recall[i])
        return ret

    def clear_metrics(self):
        self.rpn_acc.clear()
        self.rpn_metrics.clear()
        self.rpn_cls_loss.clear()
        self.rpn_loc_loss.clear()
        self.rpn_total_loss.clear()

    @staticmethod
    def convert_norm_to_float(net):
        '''
        BatchNorm layers to have parameters in single precision.
        Find all layers and convert them back to float. This can't
        be done with built in .apply as that function will apply
        fn to all modules, parameters, and buffers. Thus we wouldn't
        be able to guard the float conversion based on the module type.
        '''
        if isinstance(net, paddle.nn.layer.batchnorm._BatchNorm):
            net.float()
        for child in net.children():
            VoxelNet.convert_norm_to_float(child)
        return net


def add_sin_difference(boxes1, boxes2, boxes1_rot, boxes2_rot, factor=1.0):
    if factor != 1.0:
        boxes1_rot = factor * boxes1_rot
        boxes2_rot = factor * boxes2_rot
    rad_pred_encoding = paddle.sin(boxes1_rot) * paddle.cos(boxes2_rot)
    rad_tg_encoding = paddle.cos(boxes1_rot) * paddle.sin(boxes2_rot)
    boxes1 = paddle.concat([boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]],
                       axis=-1)
    boxes2 = paddle.concat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                       axis=-1)
    return boxes1, boxes2


def create_loss(loc_loss_ftor,
                cls_loss_ftor,
                box_preds,
                cls_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                num_class,
                encode_background_as_zeros=True,
                encode_rad_error_by_sin=True,
                sin_error_factor=1.0,
                box_code_size=7,
                num_direction_bins=2):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.reshape((batch_size, -1, box_code_size))
    if encode_background_as_zeros:
        cls_preds = cls_preds.reshape((batch_size, -1, num_class))
    else:
        cls_preds = cls_preds.reshape((batch_size, -1, num_class + 1))
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = torchplus.nn.one_hot(
        cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    global loss_flag
    if loss_flag == 0:
        torch_one_hot = np.load("torch_one_hot_targets.npy")
        if np.allclose(one_hot_targets.numpy(), torch_one_hot, atol=1e-5, rtol=1e-5) is False:
            one_hot_targets = paddle.to_tensor(torch_one_hot)
            print("compare one_hot failed..")
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        # sin(a - b) = sinacosb-cosasinb
        # reg_tg_rot = box_torch_ops.limit_period(
        #     reg_targets[..., 6:7], 0.5, 2 * np.pi / num_direction_bins)
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets, box_preds[..., 6:7], reg_targets[..., 6:7],
                                                    sin_error_factor)


    loc_losses = loc_loss_ftor(
        box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
    return loc_losses, cls_losses


def prepare_loss_weights(labels,
                         pos_cls_weight=1.0,
                         neg_cls_weight=1.0,
                         loss_norm_type=LossNormType.NormByNumPositives,
                         dtype=paddle.float32):
    """get cls_weights and reg_weights from labels.
    """
    cared = labels >= 0
    # cared: [N, num_anchors]
    positives = labels > 0
    negatives = labels == 0
    negative_cls_weights = paddle.cast(negatives, dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * paddle.cast(positives, dtype)
    reg_weights = paddle.cast(positives, dtype)
    if loss_norm_type == LossNormType.NormByNumExamples:
        num_examples = paddle.cast(cared, dtype).sum(1, keepdim=True)
        num_examples = paddle.clip(num_examples, min=1.0)
        cls_weights /= num_examples
        bbox_normalizer = paddle.cast(positives.sum(1, keepdim=True), dtype)
        reg_weights /= paddle.clip(bbox_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss
        pos_normalizer = paddle.cast(positives.sum(1, keepdim=True), dtype)
        reg_weights /= paddle.clip(pos_normalizer, min=1.0)
        cls_weights /= paddle.clip(pos_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPosNeg:
        pos_neg = paddle.cast(paddle.stack([positives, negatives], dim=-1), dtype)
        normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
        cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
        cls_normalizer = paddle.clip(cls_normalizer, min=1.0)
        # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
        normalizer = paddle.clip(normalizer, min=1.0)
        reg_weights /= normalizer[:, 0:1, 0]
        cls_weights /= cls_normalizer
    elif loss_norm_type == LossNormType.DontNorm:  # support ghm loss
        pos_normalizer = paddle.cast(positives.sum(1, keepdim=True), dtype)
        reg_weights /= paddle.clip(pos_normalizer, min=1.0)
    else:
        raise ValueError(
            f"unknown loss norm type. available: {list(LossNormType)}")
    return cls_weights, reg_weights, cared


def assign_weight_to_each_class(labels,
                                weight_per_class,
                                norm_by_num=True,
                                dtype=paddle.float32):
    weights = paddle.zeros(labels.shape, dtype=dtype)
    for label, weight in weight_per_class:
        positives = paddle.cast((labels == label), dtype)
        weight_class = weight * positives
        if norm_by_num:
            normalizer = positives.sum()
            normalizer = paddle.clip(normalizer, min=1.0)
            weight_class /= normalizer
        weights += weight_class
    return weights


def get_direction_target(anchors,
                         reg_targets,
                         one_hot=True,
                         dir_offset=0,
                         num_bins=2):
    batch_size = reg_targets.shape[0]
    anchors = anchors.reshape((batch_size, -1, anchors.shape[-1]))
    rot_gt = reg_targets[..., 6] + anchors[..., 6]
    offset_rot = box_paddle_ops.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
    dir_cls_targets = paddle.floor(offset_rot / (2 * np.pi / num_bins))
    dir_cls_targets = paddle.cast(dir_cls_targets, paddle.int64)
    dir_cls_targets = paddle.clip(dir_cls_targets, min=0, max=num_bins - 1)
    if one_hot:
        dir_cls_targets = torchplus.nn.one_hot(
            dir_cls_targets, num_bins, dtype=anchors.dtype)
    return dir_cls_targets

#output_shape =  [1, 40, 1600, 1408, 16]
#num_class =  1
#num_input_features =  4
#vfe_class_name =  SimpleVoxel
#middle_class_name =  SpMiddleFHD
#rpn_class_name =  RPNV2

#voxel
#voxels.shape= torch.Size([129274, 5, 4]) dtype= torch.float32
#num_points
#voxels.shape= torch.Size([129274]) dtype= torch.int32
#anchors
#voxels.shape= torch.Size([8, 70400, 7]) dtype= torch.float32
#labels
#voxels.shape= torch.Size([8, 70400]) dtype= torch.int32
#reg_targets
#voxels.shape= torch.Size([8, 70400, 7]) dtype= torch.float32
#importance
#voxels.shape= torch.Size([8, 70400]) dtype= torch.float32

from paddle.fluid.framework import _test_eager_guard
def test():
    with _test_eager_guard():
        output_shape =  [1, 40, 1600, 1408, 16]
        voxelnet = VoxelNet(output_shape, vfe_class_name="SimpleVoxel", middle_class_name='SpMiddleFHD', rpn_class_name='RPNV2')

        example = {}
        num_voxels = 129274
        voxel_shape = [num_voxels, 5, 4]
        voxel = paddle.randn(voxel_shape)
        num_points = paddle.randint((num_voxels))
        anchors = paddle.randn((8, 70400, 7))
        labels = paddle.zeros((8, 70400))
        reg_targets = paddle.ones((8, 70400, 7))
        importance = paddle.randn((8, 70400))

        example['voxel'] = voxel
        example['num_points'] = num_points 
        example['anchors'] = anchors
        example['labels'] = labels
        example['reg_targets'] = reg_targets
        example['importance'] = importance

        out = voxelnet(example)

#test()
