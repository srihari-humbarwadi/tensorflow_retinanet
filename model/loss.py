import tensorflow as tf


class Loss():
    def __init__(self, num_classes=None):
        self.smooth_l1 = tf.losses.Huber()

    def focal_loss(self, y_true, y_pred, alpha=0.25, gamma=2):
        y_true = tf.one_hot(y_true, depth=self.num_classes + 1)
        y_true = y_true[:, 1:]
        y_pred = tf.sigmoid(y_pred)

        CE = tf.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
        at = alpha * y_true + (1 - y_true) * (1 - alpha)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        loss = at * tf.pow(1 - pt, gamma) * tf.expand_dims(CE, axis=1)
        loss = tf.reduce_mean(loss)
        return loss

    def loss(self,
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

        Lreg = self.smooth_l1(regression_targets_positive,
                              regression_predictions_positive)
        Lcls = self.focal_loss(classification_targets_positive,
                               classification_predictions_positive)
        return Lreg / num_positive_detections, Lcls / num_positive_detections
