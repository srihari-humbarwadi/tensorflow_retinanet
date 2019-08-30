import tensorflow as tf


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
        return Lcls, Lreg, tf.reduce_mean(num_positive_detections)


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
