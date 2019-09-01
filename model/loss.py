import tensorflow as tf


class LossV2():
    def __init__(self, batch_size=None, n_classes=None):
        self.num_classes = n_classes
        self.global_batch_size = batch_size

    def focal_loss(self, y_true, y_pred, alpha=0.25, gamma=2):
        y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=self.num_classes + 1)
        y_true = y_true[:, :, 1:]
        y_pred = tf.sigmoid(y_pred)

        at = alpha * y_true + (1 - y_true) * (1 - alpha)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        f_loss = -at * tf.pow(1 - pt, gamma) * tf.math.log(pt)
        return f_loss
    
    def smooth_l1(self, y_true, y_pred, sigma=3.0):
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        sigma = tf.cast(sigma, dtype=y_pred.dtype)
        x = y_true - y_pred
        abs_x = tf.abs(x)
        sigma_squared = tf.square(sigma)
        quadratic = 0.5 * tf.square(sigma * x)
        linear = abs_x - (0.5 / sigma_squared)
        smooth_l1_loss = tf.where(tf.less(abs_x, 1./sigma_squared), quadratic, linear)
        return smooth_l1_loss

    def __call__(self,
                 classification_targets,
                 classification_predictions,
                 regression_targets,
                 regression_predictions,
                 background_mask,
                 ignore_mask):
        background_mask = tf.cast(background_mask, dtype=tf.bool)
        ignore_mask = tf.cast(ignore_mask, dtype=tf.bool)
        
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