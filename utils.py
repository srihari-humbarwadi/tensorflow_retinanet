import json
import numpy as np
import tensorflow as tf


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
        anchor_shapes['P{}'.format(int(np.log2(np.sqrt(area) // 4)))] = np.array(
            anchor_shapes['P{}'.format(int(np.log2(np.sqrt(area) // 4)))])
    return anchor_shapes


def get_anchors(input_shape=512, tensor=True):
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

    lu = tf.maximum(boxes1_t[:, None, :2], boxes2_t[:, :2])  # ld ru ??
    rd = tf.minimum(boxes1_t[:, None, 2:], boxes2_t[:, 2:])

    intersection = tf.maximum(0.0, rd - lu)
    inter_square = intersection[:, :, 0] * intersection[:, :, 1]

    square1 = boxes1[:, 2] * boxes1[:, 3]
    square2 = boxes2[:, 2] * boxes2[:, 3]

    union_square = tf.maximum(square1[:, None] + square2 - inter_square, 1e-10)
    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


def change_box_format(boxes, return_format='xywh'):
    boxes = tf.cast(boxes, dtype=tf.float32)
    if return_format == 'xywh':
        # x1 y1 x2 y2
        # 0  1  2  3
        return tf.stack([(boxes[..., 2] + boxes[..., 0]) / 2.0,
                         (boxes[..., 3] + boxes[..., 1]) / 2.0,
                         boxes[..., 2] - boxes[..., 0],
                         boxes[..., 3] - boxes[..., 1]], axis=-1)
    elif return_format == 'x1y1x2y2':
        # x  y  w  h
        # 0  1  2  3
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
    # To do, set colors for each class
    colors = tf.random.uniform(maxval=1, shape=[bbox_list.shape[0], 3])
    return tf.image.convert_image_dtype(tf.image.draw_bounding_boxes(image[None, ...],
                                                                     bboxes[None, ...],
                                                                     colors)[0, ...], dtype=tf.uint8)


def get_label(label_path, class_map, input_shape=512):
    with open(label_path, 'r') as f:
        temp = json.load(f)
    bbox = []
    class_ids = []
    for obj in temp['frames'][0]['objects']:
        if 'box2d' in obj:
            x1 = obj['box2d']['x1']
            y1 = obj['box2d']['y1']
            x2 = obj['box2d']['x2']
            y2 = obj['box2d']['y2']
            bbox.append(np.array([x1, y1, x2, y2]))
            class_ids.append(class_map[obj['category']])
    bbox = np.array(bbox, dtype=np.float32)
    H, W = 720, 1280.  # ToDO remove hardcoded values
    bbox[:, 0] = bbox[:, 0] / W
    bbox[:, 2] = bbox[:, 2] / W
    bbox[:, 1] = bbox[:, 1] / H
    bbox[:, 3] = bbox[:, 3] / H
    bbox = np.int32(bbox * input_shape)
    class_ids = np.array(class_ids, dtype=np.float32)[..., None]
    return np.concatenate([bbox, class_ids], axis=-1)


@tf.function
def get_image(image_path, input_shape=None):
    H = W = input_shape
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, size=[H, W])
    img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
    return img


def load_data(input_shape=None):
    def load_data(image_path, label):
        images = get_image(image_path, input_shape=input_shape)
        targets = encode_targets(label, input_shape=input_shape)
        # To-do : transform bbox to account image resizing, add random_flip
        return images, targets
    return load_data


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
        regression targets into one-hot encoding (N, #anchors, n_classes + 1)
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
    classification_targets = selected_gt_class_ids - \
        tf.cast(
            ignore_mask, dtype=tf.float32)
    regression_targets = tf.stack([
        (selected_gt_boxes[:, 0] - anchors[:, 0]) / anchors[:, 2],
        (selected_gt_boxes[:, 1] - anchors[:, 1]) / anchors[:, 3],
        tf.math.log(selected_gt_boxes[:, 2] / anchors[:, 2]),
        tf.math.log(selected_gt_boxes[:, 3] / anchors[:, 3])
    ], axis=-1)
    regression_targets = regression_targets * scale_factors
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

    '''gt targets are in one hot form, no need to apply  sigmoid to check correctness, use sigmoid during actual inference'''
    confidence_scores = tf.reduce_max(classification_outputs, axis=-1)
    class_ids = tf.argmax(classification_outputs, axis=-1)
    
 #     confidence_scores = tf.reduce_max(  
#         tf.nn.sigmoid(classification_outputs), axis=-1)   
    regression_outputs = regression_outputs / scale_factors
    boxes = tf.concat([(regression_outputs[:, :2] * anchors[:, 2:] + anchors[:, :2]),
                       tf.math.exp(regression_outputs[:, 2:]) * anchors[:, 2:]
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

    matched_anchors = tf.gather(anchors, tf.where(confidence_scores > classification_threshold)[:, 0])
    matched_anchors = tf.cast(change_box_format(matched_anchors, return_format='x1y1x2y2'),
                          dtype=tf.int32)    
    return final_boxes, final_class_ids, final_scores, matched_anchors