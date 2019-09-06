from model.retinanet import RetinaNet
import tensorflow as tf
from utils import encode_targets, random_image_augmentation, flip_data

strategy = tf.distribute.MirroredStrategy()

print('Tensorflow', tf.__version__)
print('Num Replicas : {}'.format(strategy.num_replicas_in_sync))

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

INPUT_SHAPE = 640
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
N_CLASSES = len(class_map)
EPOCHS = 200
training_steps = 70000 // BATCH_SIZE
val_steps = 10000 // BATCH_SIZE
LR = strategy.num_replicas_in_sync * 1e-4


with strategy.scope():
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'xmins': tf.io.VarLenFeature(tf.float32),
        'ymins': tf.io.VarLenFeature(tf.float32),
        'xmaxs': tf.io.VarLenFeature(tf.float32),
        'ymaxs': tf.io.VarLenFeature(tf.float32),
        'labels': tf.io.VarLenFeature(tf.float32)
    }

    @tf.function
    def parse_example(example_proto):
        parsed_example = tf.io.parse_single_example(
            example_proto, feature_description)
        image = tf.image.decode_jpeg(parsed_example['image'], channels=3)
        bboxes = tf.stack([
            tf.sparse_tensor_to_dense(parsed_example['xmins']),
            tf.sparse_tensor_to_dense(parsed_example['ymins']),
            tf.sparse_tensor_to_dense(parsed_example['xmaxs']),
            tf.sparse_tensor_to_dense(parsed_example['ymaxs'])
        ], axis=-1)
        class_ids = tf.reshape(tf.sparse_tensor_to_dense(
            parsed_example['labels']), [-1, 1])
        return image, bboxes, class_ids

    def load_data(input_shape):
        h, w = input_shape, input_shape

        @tf.function
        def load_data_(example_proto, input_shape=input_shape):
            image, boxes_, class_ids = parse_example(example_proto)
            image = tf.image.resize(image, size=[h, w])
            boxes = tf.stack([
                tf.clip_by_value(boxes_[:, 0] * w, 0, w),
                tf.clip_by_value(boxes_[:, 1] * h, 0, h),
                tf.clip_by_value(boxes_[:, 2] * w, 0, w),
                tf.clip_by_value(boxes_[:, 3] * h, 0, h)
            ], axis=-1)
            image, boxes = flip_data(image, boxes, w)
            image = random_image_augmentation(image)
            image = image[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
            label = tf.concat([boxes, class_ids], axis=-1)
            cls_targets, reg_targets, bg, ig = encode_targets(
                label, input_shape=input_shape)
            bg = tf.cast(bg, dtype=tf.float32)
            ig = tf.cast(ig, dtype=tf.float32)
            cls_targets = tf.cast(cls_targets, dtype=tf.float32)
            return (image, cls_targets, reg_targets, bg, ig), (tf.ones((1, )), tf.ones((1, )))
        return load_data_

    train_files = tf.data.Dataset.list_files(
        'BDD100k/train*')
    train_dataset = train_files.interleave(tf.data.TFRecordDataset,
                                           cycle_length=16,
                                           block_length=16,
                                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.map(
        load_data(INPUT_SHAPE), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(
        BATCH_SIZE, drop_remainder=True).repeat()
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_files = tf.data.Dataset.list_files(
        'BDD100k/validation*')
    val_dataset = val_files.interleave(tf.data.TFRecordDataset,
                                       cycle_length=16,
                                       block_length=16,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.map(
        load_data(INPUT_SHAPE), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True).repeat()
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)


with strategy.scope():
    model = RetinaNet(input_shape=INPUT_SHAPE,
                      n_classes=N_CLASSES, training=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=0.001)

    loss_dict = {
        'box': lambda x, y: y,
        'focal': lambda x, y: y
    }
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir='logs', update_freq='epoch'),
        tf.keras.callbacks.ModelCheckpoint('model_files/weights',
                                           save_weights_only=True,
                                           save_best_only=True,
                                           monitor='loss',
                                           verbose=1)
    ]
    model.compile(optimizer=optimizer, loss=loss_dict)
    model.fit(train_dataset,
              epochs=EPOCHS,
              steps_per_epoch=training_steps,
              validation_data=val_dataset,
              validation_steps=val_steps,
              validation_freq=5,
              callbacks=callback_list)
