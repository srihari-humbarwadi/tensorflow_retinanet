from glob import glob
from model.retinanet import RetinaNet
from model.loss import Loss
import numpy as np
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
BATCH_SIZE = 2
EPOCHS = 200
training_steps = len(train_image_paths) // BATCH_SIZE
validation_steps = len(validation_image_paths) // BATCH_SIZE


def train_data_generator():
    for i in range(len(train_image_paths)):
        yield train_image_paths[i], train_labels[i]


def validation_data_generator():
    for i in range(len(validation_image_paths)):
        yield validation_image_paths[i], validation_labels[i]


def input_fn(training=True):
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


model = RetinaNet(input_shape=input_shape, n_classes=n_classes)
model.build([None, input_shape, input_shape, 3])
loss_fn = Loss(n_classes=n_classes)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1e-3)
start_epoch = tf.Variable(0)
model_dir = 'model_files/'
checkpoint = tf.train.Checkpoint(model=model,
                                 optimizer=optimizer,
                                 start_epoch=start_epoch)
checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                directory=model_dir,
                                                max_to_keep=3)
model.summary()


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




latest_checkpoint = checkpoint_manager.latest_checkpoint
checkpoint.restore(latest_checkpoint)
s = start_epoch.numpy()
if not latest_checkpoint:
    print('Training from scratch')
else:
    print('Resuming training from epoch {}'.format(s.numpy()))


for ep in range(int(s), EPOCHS):
    for step, batch in enumerate(input_fn(training=True)()):
        Lreg, Lcls, total_loss, num_positive_detections = training_step(batch)
        logs = {
            'epoch': '{}/{}'.format(ep + 1, EPOCHS),
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
            'epoch': '{}/{}'.format(ep + 1, EPOCHS),
            'val_step': '{}/{}'.format(step + 1, validation_steps),
            'box_loss': np.round(Lreg.numpy(), 2),
            'cls_loss': np.round(Lcls.numpy(), 2),
            'total_loss': np.round(total_loss.numpy(), 2),
            'matches': np.int32(num_positive_detections.numpy())
        }
        if (step + 1) % 25 == 0:
            print(logs)
    start_epoch.assign_add(1)
    checkpoint_manager.save(checkpoint_number=ep + 1)
