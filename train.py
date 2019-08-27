from glob import glob
from model.retinanet import RetinaNet
from model.loss import Loss
import os
import tensorflow as tf
from tqdm import tqdm
from utils import get_label, load_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print(os.getcwd())
image_paths = sorted(glob('../../bdd/bdd100k/images/100k/train/*'))
label_paths = sorted(glob('../../bdd/bdd100k/labels/100k/train/*'))
print('Found {} images'.format(len(image_paths)))
print('Found {} labels'.format(len(label_paths)))

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
for image, label in zip(image_paths, label_paths):
    assert image.split(
        '/')[-1].split('.')[0] == label.split('/')[-1].split('.')[0]
labels = []
for path in tqdm(label_paths):
    labels.append(get_label(path, class_map))


n_classes = len(class_map)


def data_generator():
    for i in range(len(image_paths)):
        yield image_paths[i], labels[i]


dataset = tf.data.Dataset.from_generator(
    data_generator, output_types=(tf.string, tf.float32))
dataset = dataset.shuffle(1000)
dataset = dataset.map(
    load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.batch(16)


model = RetinaNet(H=512, W=512, n_classes=n_classes)
loss_fn = Loss(n_classes=n_classes)


for batch in dataset.take(1):
    print(batch)
