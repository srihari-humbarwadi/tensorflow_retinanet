import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def imshow(image):
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(image)


def compute_anchor_dimensions(ratios=[0.5, 1, 2],
                              scales=[1, 1.25, 1.58],
                              areas=[32 * 32, 64 * 64, 128 * 128, 256 * 256, 512 * 512]):
    anchor_shapes = {'P{}'.format(i): [] for i in range(3, 8)}
    for area in areas:
        for ratio in ratios:
            a_h = np.sqrt(area / ratio)
            a_w = area / a_h
            for scale in scales:
                h = np.int32(scale * a_h)
                w = np.int32(scale * a_w)
                anchor_shapes['P{}'.format(
                    int(np.log2(np.sqrt(area) // 4)))].append([w, h])
        anchor_shapes['P{}'.format(int(np.log2(np.sqrt(area) // 4)))] = \
            np.array(anchor_shapes['P{}'.format(
                int(np.log2(np.sqrt(area) // 4)))])
    return anchor_shapes


def get_anchors(input_shape=None, tensor=True):
    anchor_dimensions = compute_anchor_dimensions()
    anchors = []
    for i in range(3, 8):
        feature_name = 'P{}'.format(i)
        stride = 2**i
        feature_size = (input_shape) // stride

        dims = anchor_dimensions[feature_name]
        dims = dims[None, None, ...]
        dims = np.tile(dims, reps=[feature_size, feature_size, 1, 1])

        rx = (np.arange(feature_size) + 0.5) * (stride)
        ry = (np.arange(feature_size) + 0.5) * (stride)
        sx, sy = np.meshgrid(rx, ry)
        cxy = np.stack([sx, sy], axis=-1)
        cxy = cxy[:, :, None, :]
        cxy = np.tile(cxy, reps=[1, 1, 9, 1])
        anchors.append(np.reshape(
            np.concatenate([cxy, dims], axis=-1), [-1, 4]))
    anchors = np.concatenate(anchors, axis=0)
    if tensor:
        anchors = tf.constant(anchors, dtype=tf.float32)
    return anchors


@tf.function()
def compute_iou(boxes1, boxes2):
    boxes1 = tf.cast(boxes1, dtype=tf.float32)
    boxes2 = tf.cast(boxes2, dtype=tf.float32)

    boxes1_t = change_box_format(boxes1, return_format='x1y1x2y2')
    boxes2_t = change_box_format(boxes2, return_format='x1y1x2y2')

    lu = tf.maximum(boxes1_t[:, None, :2], boxes2_t[:, :2])
    rd = tf.minimum(boxes1_t[:, None, 2:], boxes2_t[:, 2:])

    intersection = tf.maximum(0.0, rd - lu)
    inter_square = intersection[:, :, 0] * intersection[:, :, 1]

    square1 = boxes1[:, 2] * boxes1[:, 3]
    square2 = boxes2[:, 2] * boxes2[:, 3]

    union_square = tf.maximum(
        square1[:, None] + square2 - inter_square, 1e-10)
    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


def change_box_format(boxes, return_format='xywh'):
    boxes = tf.cast(boxes, dtype=tf.float32)
    if return_format == 'xywh':

        return tf.stack([(boxes[..., 2] + boxes[..., 0]) / 2.0,
                         (boxes[..., 3] + boxes[..., 1]) / 2.0,
                         boxes[..., 2] - boxes[..., 0],
                         boxes[..., 3] - boxes[..., 1]], axis=-1)
    elif return_format == 'x1y1x2y2':

        return tf.stack([boxes[..., 0] - boxes[..., 2] / 2.0,
                         boxes[..., 1] - boxes[..., 3] / 2.0,
                         boxes[..., 0] + boxes[..., 2] / 2.0,
                         boxes[..., 1] + boxes[..., 3] / 2.0], axis=-1)
    return 'You should not be here'


def draw_bboxes(image, bbox_list):
    image = image / 255.
    h, w = image.shape.as_list()[:2]
    bboxes = tf.cast(tf.stack([
        bbox_list[:, 1] / h, bbox_list[:, 0] /
        w, bbox_list[:, 3] / h, bbox_list[:, 2] / w
    ], axis=-1), dtype=tf.float32)

    colors = tf.random.uniform(maxval=1, shape=[bbox_list.shape[0], 3])
    return tf.image.convert_image_dtype(tf.image.draw_bounding_boxes(image[None, ...],
                                                                     bboxes[None, ...],
                                                                     colors)[0, ...], dtype=tf.uint8)


def draw_boxes_cv2(image, bbox_list, class_ids, scores, model_input_shape, classes):
    img = np.uint8(image).copy()
    bbox_list = np.array(bbox_list, dtype=np.int32)
    h, w = img.shape[:2]
    h_scale, w_scale = h / model_input_shape, w / model_input_shape
    bbox_list = np.int32(bbox_list * np.array([w_scale, h_scale] * 2))
    for box, cls_, score in zip(bbox_list, class_ids, scores):
        text = classes[cls_] + '' + str(np.round(score, 2))
        text_orig = (box[0] + 2, box[1] + 12)
        text_bg_xy1 = (box[0], box[1])
        text_bg_xy2 = (box[0] + 60, box[1] + 18)
        img = cv2.rectangle(img, text_bg_xy1,
                            text_bg_xy2, [255, 252, 193], -1)
        img = cv2.putText(img, text, text_orig, cv2.FONT_HERSHEY_COMPLEX_SMALL, .6, [
                          0, 0, 0], 2, lineType=cv2.LINE_AA)
        img = cv2.putText(img, text, text_orig, cv2.FONT_HERSHEY_COMPLEX_SMALL, .6, [
                          255, 255, 255], 1, lineType=cv2.LINE_AA)
        img = cv2.rectangle(img, (box[0], box[1]),
                            (box[2], box[3]), [30, 15, 200], 1)
    return img

@tf.function
def random_image_augmentation(img):
    img = tf.image.random_brightness(img, max_delta=50.)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.clip_by_value(img, 0, 255)
    return img

@tf.function
def flip_data(image, boxes, w):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack([
            w - boxes[:, 2],
            boxes[:, 1],
            w - boxes[:, 0],
            boxes[:, 3]
        ], axis=-1)
    return image, boxes
        
@tf.function
def encode_targets(label, input_shape=None):
    """We use the assignment rule from RPN.
        Faster RCNN box coder follows the coding schema described below:
            ty = (y - ya) / ha
            tx = (x - xa) / wa
            th = log(h / ha)
            tw = log(w / wa)
        where x, y, w, h denote the box's center coordinates, width and height
        respectively. Similarly, xa, ya, wa, ha denote the anchor's center
        coordinates, width and height. tx, ty, tw and th denote the
        anchor-encoded center, width and height respectively.
        The open-source implementation recommends using [10.0, 10.0, 5.0, 5.0] as
        scale factors.
        See http://arxiv.org/abs/1506.01497 for details. 
        Set achors with iou < 0.5 to 0 and
        set achors with iou iou > 0.4 && < 0.5 to -1. Convert
        regression targets into one-hot encoding (N, 
        in loss_fn and exclude background class in loss calculation.
        Use [0, 0, 0, ... 0, n_classes] (all units set to zeros) to represent
        background class.
    """
    scale_factors = tf.constant([10.0, 10.0, 5.0, 5.0])
    anchors = get_anchors(input_shape=input_shape, tensor=True)
    gt_boxes = label[:, :4]
    gt_boxes = change_box_format(gt_boxes, return_format='xywh')
    gt_class_ids = label[:, 4]
    ious = compute_iou(anchors, gt_boxes)

    max_ious = tf.reduce_max(ious, axis=1)
    max_ids = tf.argmax(ious, axis=1, output_type=tf.int32)

    background_mask = max_ious > 0.5
    ignore_mask = tf.logical_and(max_ious > 0.4, max_ious < 0.5)

    selected_gt_boxes = tf.gather(gt_boxes, max_ids)
    selected_gt_class_ids = 1. + tf.gather(gt_class_ids, max_ids)

    selected_gt_class_ids = selected_gt_class_ids * \
        tf.cast(background_mask, dtype=tf.float32)
    classification_targets = selected_gt_class_ids - tf.cast(
        ignore_mask, dtype=tf.float32)
    regression_targets = tf.stack([
        (selected_gt_boxes[:, 0] - anchors[:, 0]) / anchors[:, 2],
        (selected_gt_boxes[:, 1] - anchors[:, 1]) / anchors[:, 3],
        tf.math.log(selected_gt_boxes[:, 2] / anchors[:, 2]),
        tf.math.log(selected_gt_boxes[:, 3] / anchors[:, 3])
    ], axis=-1)
    regression_targets = regression_targets * scale_factors
    reg_zeros = tf.zeros_like(regression_targets)

    '''dirty hack to filter inf occuring during box encoding
    TODO - Handle objects with small area during tfrecord generation
    '''
    regression_targets = tf.where(tf.math.is_finite(regression_targets),
                                  regression_targets,
                                  reg_zeros)

    nan_losses_filter = tf.cast(tf.reduce_prod(tf.cast(tf.math.is_finite(regression_targets),
                                                       dtype=tf.float32), axis=-1), dtype=tf.bool)
    background_mask = tf.logical_and(background_mask, nan_losses_filter)
    ignore_mask = tf.logical_and(ignore_mask, nan_losses_filter)
    return (tf.cast(classification_targets, dtype=tf.int32),
            regression_targets,
            background_mask,
            ignore_mask)


def decode_targets(classification_outputs,
                   regression_outputs,
                   input_shape=512,
                   classification_threshold=0.05,
                   nms_threshold=0.5):
    scale_factors = tf.constant([10.0, 10.0, 5.0, 5.0])
    anchors = get_anchors(input_shape=input_shape, tensor=True)

    class_ids = tf.argmax(classification_outputs, axis=-1)

    confidence_scores = tf.reduce_max(
        tf.nn.sigmoid(classification_outputs), axis=-1)
    regression_outputs = regression_outputs / scale_factors
    boxes = tf.concat([(regression_outputs[:, :2] * anchors[:, 2:] + anchors[:, :2]),
                       tf.math.exp(
                           regression_outputs[:, 2:]) * anchors[:, 2:]
                       ], axis=-1)
    boxes = change_box_format(boxes, return_format='x1y1x2y2')

    nms_indices = tf.image.non_max_suppression(boxes,
                                               confidence_scores,
                                               score_threshold=classification_threshold,
                                               iou_threshold=nms_threshold,
                                               max_output_size=200)
    final_class_ids = tf.gather(class_ids, nms_indices)
    final_scores = tf.gather(confidence_scores, nms_indices)
    final_boxes = tf.cast(tf.gather(boxes, nms_indices), dtype=tf.int32)

    matched_anchors = tf.gather(anchors, tf.where(
        confidence_scores > classification_threshold)[:, 0])
    matched_anchors = tf.cast(change_box_format(matched_anchors, return_format='x1y1x2y2'),
                              dtype=tf.int32)
    return final_boxes, final_class_ids, final_scores, matched_anchors
