from resnet50 import ResNet50
import tensorflow as tf
from .blocks import build_classification_subnet, build_regression_subnet, conv_block, Upsampling
from .loss import LossV2
import sys
sys.path.append('..')
from utils import get_anchors


def RetinaNet(input_shape=None, n_classes=None, training=False):
    H = W = input_shape
    num_anchors = get_anchors(input_shape=H).shape[0]
    loss_fn = LossV2(n_classes=n_classes)

    base_model = ResNet50(
        input_shape=[H, W, 3], weights='imagenet', include_top=False)

    resnet_block_output_names = [
        'activation_21', 'activation_39', 'activation_48']

    resnet_block_outputs = {'C{}'.format(idx + 3): base_model.get_layer(
        layer).output for idx, layer in enumerate(resnet_block_output_names)}
    resnet_block_outputs = {level: conv_block(
        tensor, 256, 1, name=level + '_1x1') for level, tensor in resnet_block_outputs.items()}

    P5 = resnet_block_outputs['C5']
    P6 = conv_block(base_model.get_layer(
        'activation_48').output, 256, 3, strides=2, name='P6')
    P6_relu = tf.keras.layers.ReLU(name='P6')(P6)
    P7 = conv_block(P6_relu, 256, 3, strides=2, name='P7')
    M4 = tf.keras.layers.add([tf.keras.layers.Lambda(Upsampling, arguments={'scale': 2}, name='P5_UP')(
        P5), resnet_block_outputs['C4']], name='P4_merge')
    M3 = tf.keras.layers.add([tf.keras.layers.Lambda(Upsampling, arguments={'scale': 2}, name='P4_UP')(
        M4), resnet_block_outputs['C3']], name='P3_merge')
    P4 = conv_block(M4, 256, 3, name='P4')
    P3 = conv_block(M3, 256, 3, name='P3')
#         pyrammid_features = [P7, P6, P5, P4, P3]
    pyrammid_features = [P3, P4, P5, P6, P7]

    classification_subnet = build_classification_subnet(
        n_classes=n_classes)
    regression_subnet = build_regression_subnet()

    classification_outputs = [classification_subnet(
        level) for level in pyrammid_features]
    regression_outputs = [regression_subnet(
        level) for level in pyrammid_features]

    classification_head = tf.keras.layers.concatenate(
        classification_outputs, axis=1, name='classification_head')
    regression_head = tf.keras.layers.concatenate(
        regression_outputs, axis=1, name='regression_head')

    image_input = base_model.input
    classification_targets = tf.keras.layers.Input(shape=[num_anchors])
    regression_targets = tf.keras.layers.Input(shape=[num_anchors, 4])
    background_mask = tf.keras.layers.Input(shape=[num_anchors])
    ignore_mask = tf.keras.layers.Input(shape=[num_anchors])

    Lreg, Lcls, _ = tf.keras.layers.Lambda(loss_fn)([classification_targets,
                                                     classification_head,
                                                     regression_targets,
                                                     regression_head,
                                                     background_mask,
                                                     ignore_mask])

    Lreg = tf.keras.layers.Lambda(
        lambda x: tf.reshape(x, [-1, 1]), name='box')(Lreg)
    Lcls = tf.keras.layers.Lambda(
        lambda x: tf.reshape(x, [-1, 1]), name='focal')(Lcls)

    if training:
        _inputs = [image_input, classification_targets,
                   regression_targets, background_mask, ignore_mask]
        _outputs = [Lreg, Lcls]
    else:
        _inputs = [image_input]
        _outputs = [classification_head, regression_head]
    model = tf.keras.Model(inputs=_inputs, outputs=_outputs, name='RetinaNet')
    return model
