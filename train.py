from glob import glob
from model.retinanet import RetinaNet
from model.loss import Loss
import os
import tensorflow as tf
from tqdm import tqdm
from utils import get_label, load_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train_image_paths = sorted(
    glob('../../bdd/bdd100k/images/100k/train/*'))
train_label_paths = sorted(
    glob('../../bdd/bdd100k/labels/100k/train/*'))
validation_image_paths = sorted(
    glob('../../bdd/bdd100k/images/100k/val/*'))
validation_label_paths = sorted(
    glob('../../bdd/bdd100k/labels/100k/val/*'))

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

train_labels = []
validation_labels = []

for path in tqdm(train_label_paths):
    train_labels.append(get_label(path, class_map))
for path in tqdm(validation_label_paths):
    validation_labels.append(get_label(path, class_map))

n_classes = len(class_map)
input_shape = 512
BATCH_SIZE = 4
EPOCHS = 2
training_steps = len(train_image_paths) // BATCH_SIZE
validation_steps = len(validation_image_paths) // BATCH_SIZE


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
        train_dataset = train_dataset.shuffle(1000)
        train_dataset = train_dataset.map(
            load_data(input_shape), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return train_dataset.repeat()

    def validation_input_fn():
        validation_dataset = tf.data.Dataset.from_generator(
            validation_data_generator, output_types=(tf.string, tf.float32))
        validation_dataset = validation_dataset.shuffle(1000)
        validation_dataset = validation_dataset.map(
            load_data(input_shape), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_dataset = validation_dataset.batch(
            BATCH_SIZE, drop_remainder=True)
        validation_dataset = validation_dataset.prefetch(
            tf.data.experimental.AUTOTUNE)
        return validation_dataset.repeat()
    if training:
        return train_input_fn
    return validation_input_fn


model = RetinaNet(input_shape=input_shape, n_classes=n_classes)
loss_fn = Loss(n_classes=n_classes)

for ep in range(EPOCHS):
    for batch in tqdm(input_fn(training=True)(), total=training_steps):
        pass
    for batch in tqdm(input_fn(training=False)(), total=validation_steps):
        pass
