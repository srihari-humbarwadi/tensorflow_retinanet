import cv2
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os

print('Tensorflow', tf.__version__)


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

    lu = tf.maximum(boxes1_t[:, None, :2], boxes2_t[:, :2])
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


def draw_boxes_cv2(image, bbox_list):
    img = np.uint8(image).copy()
    bbox_list = np.array(bbox_list, dtype=np.int32)
    for box in bbox_list:
        img = cv2.rectangle(img, (box[0], box[1]),
                            (box[2], box[3]), [0, 0, 200], 3)
    return img


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
    H, W = 720, 1280.
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
        regression targets into one-hot encoding (N, 
        in loss_fn and exclude background class in loss calculation.
        Use [0, 0, 0, ... 0, n_classes] (all units set to zeros) to represent
        background class.
    """
    scale_factors = tf.constant([5.0, 5.0, 5.0, 5.0])
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
    return (tf.cast(classification_targets, dtype=tf.int32),
            regression_targets,
            background_mask,
            ignore_mask)


def decode_targets(classification_outputs,
                   regression_outputs,
                   input_shape=512,
                   classification_threshold=0.05,
                   nms_threshold=0.5):
    scale_factors = tf.constant([5.0, 5.0, 5.0, 5.0])
    anchors = get_anchors(input_shape=input_shape, tensor=True)

    '''gt targets are in one hot form, no need to apply  sigmoid to check correctness, use sigmoid during actual inference'''
    class_ids = tf.argmax(classification_outputs, axis=-1)

    confidence_scores = tf.reduce_max(
        tf.nn.sigmoid(classification_outputs), axis=-1)
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

    matched_anchors = tf.gather(anchors, tf.where(
        confidence_scores > classification_threshold)[:, 0])
    matched_anchors = tf.cast(change_box_format(matched_anchors, return_format='x1y1x2y2'),
                              dtype=tf.int32)
    return final_boxes, final_class_ids, final_scores, matched_anchors


def conv_block(x,
               n_filters,
               size,
               strides=1,
               kernel_init='he_normal',
               bias_init='zeros',
               bn_activated=False, name=''):
    x = tf.keras.layers.Conv2D(filters=n_filters,
                               kernel_size=size,
                               padding='same',
                               strides=strides,
                               kernel_initializer=kernel_init,
                               bias_initializer=bias_init,
                               name='conv_' + name if name else None)(x)
    if bn_activated:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    return x


def Upsampling(tensor, scale=2):
    dims = tensor.shape.as_list()[1:-1]
    return tf.image.resize(tensor, size=[dims[0] * scale, dims[1] * scale])


def build_classification_subnet(n_classes=None, n_anchors=9, p=0.01):
    input_layer = tf.keras.layers.Input(shape=[None, None, 256])
    x = input_layer
    for i in range(4):
        x = conv_block(
            x, 256, 3, kernel_init=tf.keras.initializers.RandomNormal(0.0, 0.01))
        x = tf.keras.layers.ReLU()(x)
    bias_init = -np.log((1 - p) / p)
    output_layer = tf.keras.layers.Conv2D(filters=n_classes * n_anchors,
                                          kernel_size=3,
                                          padding='same',
                                          kernel_initializer=tf.keras.initializers.RandomNormal(
                                              0.0, 0.01),
                                          bias_initializer=tf.keras.initializers.Constant(
                                              value=bias_init),
                                          activation=None)(x)
    output_layer = tf.keras.layers.Reshape(
        target_shape=[-1, n_classes])(output_layer)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer, name='classification_subnet')


def build_regression_subnet(n_anchors=9):
    input_layer = tf.keras.layers.Input(shape=[None, None, 256])
    x = input_layer
    for i in range(4):
        x = conv_block(
            x, 256, 3, kernel_init=tf.keras.initializers.RandomNormal(0.0, 0.01))
        x = tf.keras.layers.ReLU()(x)
    output_layer = tf.keras.layers.Conv2D(filters=4 * n_anchors,
                                          kernel_size=3,
                                          padding='same',
                                          kernel_initializer=tf.keras.initializers.RandomNormal(
                                              0.0, 0.01),
                                          bias_initializer=tf.keras.initializers.zeros(),
                                          activation=None)(x)
    output_layer = tf.keras.layers.Reshape(target_shape=[-1, 4])(output_layer)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer, name='regression_subnet')


class LossV2():
    def __init__(self, batch_size=None, n_classes=None):
        self.smooth_l1 = tf.losses.Huber(reduction='none')
        self.num_classes = n_classes
        self.global_batch_size = batch_size

    def focal_loss(self, y_true, y_pred, alpha=0.25, gamma=2):
        y_true = tf.one_hot(y_true, depth=self.num_classes + 1)
        y_true = y_true[:, :, 1:]
        y_pred = tf.sigmoid(y_pred)

        at = alpha * y_true + (1 - y_true) * (1 - alpha)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        f_loss = -at * tf.pow(1 - pt, gamma) * tf.math.log(pt)
        return f_loss

    def __call__(self,
                 classification_targets,
                 classification_predictions,
                 regression_targets,
                 regression_predictions,
                 background_mask,
                 ignore_mask):
        num_positive_detections = tf.maximum(tf.reduce_sum(
            tf.cast(background_mask, dtype=tf.float32), axis=-1), 1.0)

        positive_classification_mask = tf.expand_dims(
            tf.logical_not(ignore_mask), axis=-1)
        positive_classification_mask = tf.tile(
            positive_classification_mask, multiples=[1, 1, self.num_classes])

        positive_regression_mask = tf.expand_dims(background_mask, axis=-1)
        positive_regression_mask = tf.tile(
            positive_regression_mask, multiples=[1, 1, 4])

        Lcls = self.focal_loss(classification_targets,
                               classification_predictions)
        Lreg = self.smooth_l1(regression_targets, regression_predictions)
        Lcls = Lcls * tf.cast(positive_classification_mask, dtype=tf.float32)
        Lreg = Lreg * tf.cast(positive_regression_mask, dtype=tf.float32)

        Lcls = tf.reduce_sum(
            Lcls, axis=[1, 2]) / num_positive_detections
        Lreg = tf.reduce_sum(
            Lreg, axis=[1, 2]) / num_positive_detections

        Lcls = tf.nn.compute_average_loss(
            Lcls, global_batch_size=self.global_batch_size)
        Lreg = tf.nn.compute_average_loss(
            Lreg, global_batch_size=self.global_batch_size)
        return Lreg, Lcls, tf.reduce_mean(num_positive_detections)


class Loss():
    def __init__(self, n_classes=None):
        self.smooth_l1 = tf.losses.Huber(delta=0.1, reduction='none')
        self.num_classes = n_classes

    def focal_loss(self, y_true, y_pred, alpha=0.25, gamma=2):
        y_true = tf.one_hot(y_true, depth=self.num_classes + 1)
        y_true = y_true[:, 1:]
        y_pred = tf.sigmoid(y_pred)

        at = alpha * y_true + (1 - y_true) * (1 - alpha)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        f_loss = -at * tf.pow(1 - pt, gamma) * tf.math.log(pt)
        return f_loss

    def __call__(self,
                 classification_targets,
                 classification_predictions,
                 regression_targets,
                 regression_predictions,
                 background_mask,
                 ignore_mask):
        num_positive_detections = tf.maximum(tf.reduce_sum(
            tf.cast(background_mask, dtype=tf.float32)), 1.0)
        positive_classification_mask = tf.logical_not(ignore_mask)

        regression_targets_positive = tf.boolean_mask(
            regression_targets, background_mask)
        regression_predictions_positive = tf.boolean_mask(
            regression_predictions, background_mask)

        classification_targets_positive = tf.boolean_mask(
            classification_targets, positive_classification_mask)
        classification_predictions_positive = tf.boolean_mask(
            classification_predictions, positive_classification_mask)

        Lreg = tf.reduce_sum(self.smooth_l1(regression_targets_positive,
                                            regression_predictions_positive))
        Lcls = tf.reduce_sum(self.focal_loss(classification_targets_positive,
                                             classification_predictions_positive))
        return (Lreg + Lcls) / num_positive_detections


def RetinaNet(input_shape=None, n_classes=None):
    H = W = input_shape
    base_model = tf.keras.applications.ResNet50(
        input_shape=[H, W, 3], weights='imagenet', include_top=False)

    resnet_block_output_names = ['conv3_block4_out',
                                 'conv4_block6_out', 'conv5_block3_out']
    resnet_block_outputs = {'C{}'.format(idx + 3): base_model.get_layer(
        layer).output for idx, layer in enumerate(resnet_block_output_names)}
    resnet_block_outputs = {level: conv_block(
        tensor, 256, 1, name=level + '_1x1') for level, tensor in resnet_block_outputs.items()}

    P5 = resnet_block_outputs['C5']
    P6 = conv_block(base_model.get_layer(
        'conv5_block3_out').output, 256, 3, strides=2, name='P6')
    P6_relu = tf.keras.layers.ReLU(name='P6')(P6)
    P7 = conv_block(P6_relu, 256, 3, strides=2, name='P7')
    M4 = tf.keras.layers.add([tf.keras.layers.Lambda(Upsampling, arguments={'scale': 2}, name='P5_UP')(
        P5), resnet_block_outputs['C4']], name='P4_merge')
    M3 = tf.keras.layers.add([tf.keras.layers.Lambda(Upsampling, arguments={'scale': 2}, name='P4_UP')(
        M4), resnet_block_outputs['C3']], name='P3_merge')
    P4 = conv_block(M4, 256, 3, name='P4')
    P3 = conv_block(M3, 256, 3, name='P3')
    pyrammid_features = [P7, P6, P5, P4, P3]

    classification_subnet = build_classification_subnet(n_classes=n_classes)
    regression_subnet = build_regression_subnet()

    classification_outputs = [classification_subnet(
        level) for level in pyrammid_features]
    regression_outputs = [regression_subnet(
        level) for level in pyrammid_features]

    classification_head = tf.keras.layers.concatenate(
        classification_outputs, axis=1, name='classification_head')
    regression_head = tf.keras.layers.concatenate(
        regression_outputs, axis=1, name='regression_head')

    return tf.keras.Model(inputs=base_model.input, outputs=[
        classification_head, regression_head],
        name='RetinaNet')


INPUT_SHAPE = 640
BATCH_SIZE = 4
N_CLASSES = 20
EPOCHS = 200
training_steps = 2501 // BATCH_SIZE
validation_steps = 2510 // BATCH_SIZE


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


def load_data(input_shape):
    h, w = input_shape, input_shape

    @tf.function
    def load_data_(example, input_shape=input_shape):
        image = tf.cast(example['image'], dtype=tf.float32)
        boxes_ = example['objects']['bbox']
        class_ids = tf.expand_dims(
            tf.cast(example['objects']['label'], dtype=tf.float32), axis=-1)
        image = tf.image.resize(image, size=[h, w])

        boxes = tf.stack([
            tf.clip_by_value(boxes_[:, 1] * w, 0, w),
            tf.clip_by_value(boxes_[:, 0] * h, 0, h),
            tf.clip_by_value(boxes_[:, 3] * w, 0, w),
            tf.clip_by_value(boxes_[:, 2] * h, 0, h)
        ], axis=-1)
        image, boxes = flip_data(image, boxes, w)
        label = tf.concat([boxes, class_ids], axis=-1)
        cls_targets, reg_targets, bg, ig = encode_targets(
            label, input_shape=input_shape)
        return image, (cls_targets, reg_targets, bg, ig)
    return load_data_


train_dataset = tfds.load('voc2007', shuffle_files=False, split=['train'])[0]
train_dataset = train_dataset.map(load_data(
    input_shape=INPUT_SHAPE), num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tfds.load('voc2007', shuffle_files=False,
                        split=['validation'])[0]
val_dataset = val_dataset.map(load_data(
    input_shape=INPUT_SHAPE), num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
train_dataset, val_dataset


i = 0
for batch in train_dataset.take(1):
    break
image = batch[0][i]
cls_targets, reg_targets, _, _ = batch[1]
classification_outputs = tf.one_hot(cls_targets[i], depth=N_CLASSES + 1)[:, 1:]
regression_outputs = reg_targets[i]

scale_factors = tf.constant([5.0, 5.0, 5.0, 5.0])
anchors = get_anchors(input_shape=INPUT_SHAPE, tensor=True)
class_ids = tf.argmax(classification_outputs, axis=-1)
confidence_scores = tf.reduce_max(classification_outputs, axis=-1)
regression_outputs = regression_outputs / scale_factors
boxes = tf.concat([(regression_outputs[:, :2] * anchors[:, 2:] + anchors[:, :2]),
                   tf.math.exp(regression_outputs[:, 2:]) * anchors[:, 2:]
                   ], axis=-1)
boxes = change_box_format(boxes, return_format='x1y1x2y2')

nms_indices = tf.image.non_max_suppression(boxes,
                                           confidence_scores,
                                           score_threshold=0.05,
                                           iou_threshold=0.5,
                                           max_output_size=200)
final_class_ids = tf.gather(class_ids, nms_indices)
final_scores = tf.gather(confidence_scores, nms_indices)
final_boxes = tf.cast(tf.gather(boxes, nms_indices), dtype=tf.int32)

matched_anchors = tf.gather(anchors, tf.where(confidence_scores > 0.05)[:, 0])
matched_anchors = tf.cast(change_box_format(matched_anchors, return_format='x1y1x2y2'),
                          dtype=tf.int32)
img = draw_boxes_cv2(image, boxes)
imshow(img)
print(final_boxes.numpy())
print(final_class_ids.numpy())
print(final_scores.numpy())


model = RetinaNet(input_shape=INPUT_SHAPE, n_classes=N_CLASSES)
loss_fn = LossV2(batch_size=BATCH_SIZE, n_classes=N_CLASSES)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1e-3)
[x.shape for x in model(tf.random.normal((1, INPUT_SHAPE, INPUT_SHAPE, 3)))]


@tf.function
def training_step(batch):
    image, (classification_targets,
            regression_targets,
            background_mask,
            ignore_mask) = batch
    with tf.GradientTape() as tape:
        classification_predictions, regression_predictions = model(
            image, training=True)
        Lreg, Lcls, num_positive_detections = loss_fn(classification_targets,
                                                      classification_predictions,
                                                      regression_targets,
                                                      regression_predictions,
                                                      background_mask,
                                                      ignore_mask)
        total_loss = Lreg + Lcls
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return Lreg, Lcls, total_loss, num_positive_detections


@tf.function
def validation_step(batch):
    image, (classification_targets,
            regression_targets,
            background_mask,
            ignore_mask) = batch
    classification_predictions, regression_predictions = model(
        image, training=False)
    Lreg, Lcls, num_positive_detections = loss_fn(classification_targets,
                                                  classification_predictions,
                                                  regression_targets,
                                                  regression_predictions,
                                                  background_mask,
                                                  ignore_mask)
    total_loss = Lreg + Lcls
    return Lreg, Lcls, total_loss, num_positive_detections


for ep in range(0, EPOCHS):
    for step, batch in enumerate(train_dataset):
        Lreg, Lcls, total_loss, num_positive_detections = training_step(batch)
        logs = {
            'epoch': '{}/{}'.format(ep + 1, EPOCHS),
            'train_step': '{}/{}'.format(step + 1, training_steps),
            'box_loss': np.round(Lreg.numpy(), 2),
            'cls_loss': np.round(Lcls.numpy(), 2),
            'total_loss': np.round(total_loss.numpy(), 2),
            'matches': np.int32(num_positive_detections.numpy())
        }
        print(logs)
    for step, batch in enumerate(val_dataset):
        Lreg, Lcls, total_loss, num_positive_detections = validation_step(
            batch)
        logs = {
            'epoch': '{}/{}'.format(ep + 1, EPOCHS),
            'val_step': '{}/{}'.format(step + 1, validation_steps),
            'box_loss': np.round(Lreg.numpy(), 2),
            'cls_loss': np.round(Lcls.numpy(), 2),
            'total_loss': np.round(total_loss.numpy(), 2),
            'matches': np.int32(num_positive_detections.numpy())
        }
        if (step + 1) % 25 == 0:
            print(logs)
    model.save_weights('{}_epoch_weights.h5'.format(ep + 1))
