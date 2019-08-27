import tensorflow as tf
from .blocks import build_classification_subnet, build_regression_subnet, conv_block, Upsampling


def RetinaNet(input_shape=None, n_classes=None):
    H = W = input_shape
    base_model = tf.keras.applications.ResNet50(
        input_shape=[H, W, 3], weights='imagenet', include_top=False)

    resnet_block_output_names = ['conv3_block4_out',
                                 'conv4_block6_out', 'conv5_block3_out']
    resnet_block_outputs = {'C{}'.format(idx + 3): base_model.get_layer(
        layer).output for idx, layer in enumerate(resnet_block_output_names)}
    resnet_block_outputs = {level: conv_block(
        tensor, 256, 1) for level, tensor in resnet_block_outputs.items()}

    P5 = resnet_block_outputs['C5']
    P6 = conv_block(P5, 256, 3, strides=2)
    P6_relu = tf.keras.layers.ReLU()(P6)
    P7 = conv_block(P6_relu, 256, 3, strides=2)
    M4 = tf.keras.layers.add([tf.keras.layers.Lambda(Upsampling, arguments={'scale': 2})(
        P5), resnet_block_outputs['C4']])
    M3 = tf.keras.layers.add([tf.keras.layers.Lambda(Upsampling, arguments={'scale': 2})(
        M4), resnet_block_outputs['C3']])
    P4 = conv_block(M4, 256, 3)
    P3 = conv_block(M3, 256, 3)
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
