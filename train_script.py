#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install tensorflow-gpu==2.0.0-rc0
from glob import glob
import json
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm_notebook

print('TensorFlow', tf.__version__)
# !gdown --id 1quc2K7HfoEm8YRspAkTNjGaTamZu97WN
# !unzip bdd100k.zip 
get_ipython().system('nvidia-smi')
base_path = './'


# In[2]:


train_image_paths = sorted(
    glob('bdd100k/images/100k/train/*'))
train_label_paths = sorted(
    glob('bdd100k/labels/100k/train/*'))
validation_image_paths = sorted(
    glob('bdd100k/images/100k/val/*'))
validation_label_paths = sorted(
    glob('bdd100k/labels/100k/val/*'))

print('Found training {} images'.format(len(train_image_paths)))
print('Found training {} labels'.format(len(train_label_paths)))
print('Found validation {} images'.format(len(validation_image_paths)))
print('Found validation {} labels'.format(len(validation_label_paths)))

class_map = {value: idx for idx, value in enumerate(['bus',
                                                     'traffic light',
                                                     'traffic sign',
                                                     'person',
                                                     'bike',
                                                     'truck',
                                                     'motor',
                                                     'car',
                                                     'train',
                                                     'rider'])}
for image, label in zip(train_image_paths, train_label_paths):
    assert image.split(
        '/')[-1].split('.')[0] == label.split('/')[-1].split('.')[0]
for image, label in zip(validation_image_paths, validation_label_paths):
    assert image.split(
        '/')[-1].split('.')[0] == label.split('/')[-1].split('.')[0]


# In[14]:


n_classes = len(class_map)
input_shape = 512
BATCH_SIZE = 4
EPOCHS = 200
training_steps = len(train_image_paths) // BATCH_SIZE
validation_steps = len(validation_image_paths) // BATCH_SIZE


# In[4]:


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

train_labels = []
validation_labels = []

for path in tqdm_notebook(train_label_paths):
    train_labels.append(get_label(path, class_map))
for path in tqdm_notebook(validation_label_paths):
    validation_labels.append(get_label(path, class_map))


# In[8]:


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

    selected_gt_class_ids = selected_gt_class_ids *         tf.cast(background_mask, dtype=tf.float32)
    classification_targets = selected_gt_class_ids -         tf.cast(
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

    matched_anchors = tf.gather(anchors, tf.where(confidence_scores > classification_threshold)[:, 0])
    matched_anchors = tf.cast(change_box_format(matched_anchors, return_format='x1y1x2y2'),
                          dtype=tf.int32)    
    return final_boxes, final_class_ids, final_scores, matched_anchors


# In[9]:


def train_data_generator():
    for i in range(len(train_image_paths)):
        yield train_image_paths[i], train_labels[i]


def validation_data_generator():
    for i in range(len(validation_image_paths)):
        yield validation_image_paths[i], validation_labels[i]


def input_fn(training=True, context_id=None):
    def train_input_fn():
        train_dataset = tf.data.Dataset.from_generator(
            train_data_generator, output_types=(tf.string, tf.float32))
        train_dataset = train_dataset.shuffle(1024)
        train_dataset = train_dataset.map(
            load_data(input_shape), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return train_dataset

    def validation_input_fn():
        validation_dataset = tf.data.Dataset.from_generator(
            validation_data_generator, output_types=(tf.string, tf.float32))
        validation_dataset = validation_dataset.map(
            load_data(input_shape), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_dataset = validation_dataset.batch(
            BATCH_SIZE, drop_remainder=True)
        validation_dataset = validation_dataset.prefetch(
            tf.data.experimental.AUTOTUNE)
        return validation_dataset
    if training:
        return train_input_fn
    return validation_input_fn


# In[10]:


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
        return Lreg /num_positive_detections,  Lcls / num_positive_detections, num_positive_detections

      
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
        
        positive_classification_mask = tf.expand_dims(tf.logical_not(ignore_mask), axis=-1)
        positive_classification_mask = tf.tile(positive_classification_mask, multiples=[1, 1, self.num_classes])
        
        positive_regression_mask = tf.expand_dims(background_mask, axis=-1)
        positive_regression_mask = tf.tile(positive_regression_mask, multiples=[1, 1, 4])
        
        Lcls = self.focal_loss(classification_targets, classification_predictions)
        Lreg = self.smooth_l1(regression_targets, regression_predictions)
        Lcls = Lcls * tf.cast(positive_classification_mask, dtype=tf.float32)
        Lreg = Lreg * tf.cast(positive_regression_mask, dtype=tf.float32)
        
        Lcls = tf.reduce_sum(Lcls, axis=[1, 2]) / num_positive_detections
        Lreg = tf.reduce_sum(Lreg, axis=[1, 2]) / num_positive_detections
        
        Lcls  = tf.nn.compute_average_loss(Lcls, global_batch_size=self.global_batch_size)
        Lreg = tf.nn.compute_average_loss(Lreg, global_batch_size=self.global_batch_size)
        return Lreg, Lcls, tf.reduce_mean(num_positive_detections)


# In[11]:


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
    bias_init = -tf.math.log((1 - p) / p).numpy()
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


# In[12]:


model = RetinaNet(input_shape=input_shape, n_classes=n_classes)
model.build([None, input_shape, input_shape, 3])
loss_fn = LossV2(batch_size=BATCH_SIZE, n_classes=n_classes)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1e-3)
start_epoch = tf.Variable(0)
model_dir = '{}/model_files/'.format(base_path)
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, start_epoch=start_epoch)
checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                directory=model_dir,
                                                max_to_keep=3)
# model.summary()


# In[16]:


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


# In[17]:


latest_checkpoint = checkpoint_manager.latest_checkpoint
checkpoint.restore(latest_checkpoint)
s = start_epoch.numpy()
if not latest_checkpoint:
  print('Training from scratch')
else:
  print('Resuming training from epoch {}'.format(s.numpy()))

  
for ep in range(0, EPOCHS):
    for step, batch in enumerate(input_fn(training=True)()):
        Lreg, Lcls, total_loss, num_positive_detections = training_step(batch)
        logs = {
            'epoch':'{}/{}'.format(ep + 1, EPOCHS),
            'train_step': '{}/{}'.format(step + 1, training_steps),
            'box_loss': np.round(Lreg.numpy(), 2),
            'cls_loss': np.round(Lcls.numpy(), 2),
            'total_loss': np.round(total_loss.numpy(), 2),
            'matches': np.int32(num_positive_detections.numpy())
        }
        if (step + 1) % 10 == 0:
            print(logs)
    for step, batch in enumerate(input_fn(training=False)()):
        Lreg, Lcls, total_loss, num_positive_detections = validation_step(
            batch)
        logs = {
            'epoch':'{}/{}'.format(ep + 1, EPOCHS),
            'val_step': '{}/{}'.format(step + 1, validation_steps),
            'box_loss': np.round(Lreg.numpy(), 2),
            'cls_loss': np.round(Lcls.numpy(), 2),
            'total_loss': np.round(total_loss.numpy(), 2),
            'matches': np.int32(num_positive_detections.numpy())
        }
        if (step + 1) % 5 == 0:
            print(logs)
    start_epoch.assign_add(1)
    checkpoint_manager.save(checkpoint_number=ep+1)


# In[15]:


from skimage.io import imsave
image = get_image(input_shape=512, image_path=train_image_paths[0])[None, ...]
c_preds, r_preds = model(image)
b, c, s, _ = decode_targets(c_preds[0], r_preds[0], classification_threshold=.5, nms_threshold=0.5)
img = (image[0] + tf.constant([103.939, 116.779, 123.68]))[:, :, ::-1]
img = draw_bboxes(img, b).numpy()
imsave('inference.png', img)
b, c, s


# In[ ]:


ep


# In[ ]:


s


# In[ ]:




