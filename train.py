from model.retinanet import retinanet
from model.loss import Loss
import tensorflow as tf
from utils import *


image_paths = sorted(glob('../bdd/bdd100k/images/100k/train/*'))
label_paths = sorted(glob('../bdd/bdd100k/labels/100k/train/*'))
for image, label in zip(image_paths, label_paths):
    assert image.split('/')[-1].split('.')[0] == label.split('/')[-1].split('.')[0]
labels = []
for path in tqdm(label_paths):
    labels.append(get_label(path))

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
n_classes = len(class_map)

def data_generator():
    for i in range(len(image_paths)):
        yield image_paths[i], labels[i]

dataset = tf.data.Dataset.from_generator(data_generator, output_types=(tf.string, tf.float32))
dataset = dataset.shuffle(1000)
dataset = dataset.map(load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.batch(16)