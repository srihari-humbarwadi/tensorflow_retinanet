import tensorflow as tf


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


def build_classification_subnet(n_classes=100, n_anchors=9, p=0.01):
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
