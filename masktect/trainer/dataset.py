import albumentations as albu
import os
import PIL
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from .config import CFG, REPLICAS, AUTO


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.PadIfNeeded(400, 400),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        albu.Resize(320, 320),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(400, 400),
        albu.Resize(320, 320),
    ]
    return albu.Compose(test_transform)


def read_labeled_tfrecord(example):
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }           
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image'], example['label']


def read_unlabeled_tfrecord(example, return_image_name):
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image'], example['label'] if return_image_name else 0


train_augment = get_training_augmentation()
val_augment = get_validation_augmentation()


def prepare_image(img, cfg=None, is_test=False):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [cfg['read_size'], cfg['read_size']])
    img = tf.cast(img, tf.float32) / 255.0

    if not is_test:
        img = train_augment(image=img)
    else:
        img = val_augment(image=img)

    img = tf.image.resize(img, [cfg['net_size'], cfg['net_size']])
    img = tf.reshape(img, [cfg['net_size'], cfg['net_size'], 3])
    return img


def get_dataset(files, cfg, is_test=False, shuffle=False, repeat=False,
                labeled=True, return_image_names=True):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(1024 * 8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    if labeled:
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    else:
        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_names),
                    num_parallel_calls=AUTO)

    ds = ds.map(lambda img, imgname_or_label: (prepare_image(img, is_test=is_test, cfg=cfg),
                                               imgname_or_label),
                num_parallel_calls=AUTO)

    ds = ds.batch(cfg['batch_size'] * REPLICAS)
    ds = ds.prefetch(AUTO)
    return ds


def show_dataset(thumb_size, cols, rows, ds):
    mosaic = PIL.Image.new(mode='RGB', size=(thumb_size * cols + (cols - 1),
                                             thumb_size * rows + (rows - 1)))

    for idx, data in enumerate(iter(ds)):
        img, target_or_imgid = data
        ix = idx % cols
        iy = idx // cols
        img = np.clip(img.numpy() * 255, 0, 255).astype(np.uint8)
        img = PIL.Image.fromarray(img)
        img = img.resize((thumb_size, thumb_size), resample=PIL.Image.BILINEAR)
        mosaic.paste(img, (ix * thumb_size + ix,
                           iy * thumb_size + iy))

    plt.show(mosaic)


np.random.seed(42)
DATASET_PATH = "data/images"
WITH = "with_mask"
WITHOUT = "without_mask"
with_mask = os.listdir(f"{DATASET_PATH}/{WITH}")
without_mask = os.listdir(f"{DATASET_PATH}/{WITHOUT}")
# list of (path, label)
all_files_with_labels = [(f"{DATASET_PATH}/{WITH}/{x}", 1) for x in with_mask] + \
                            [(f"{DATASET_PATH}/{WITHOUT}/{x}", 0) for x in without_mask]
np.random.shuffle(all_files_with_labels)
TRAIN_RATIO = 0.8
TRAIN_COUNT = int(0.8 * len(all_files_with_labels))
files_train = all_files_with_labels[:TRAIN_COUNT]
files_test = [x for x, _ in all_files_with_labels[TRAIN_COUNT:]]


def main():
    ds = get_dataset(files_train, CFG).unbatch().take(12 * 5)
    show_dataset(64, 12, 5, ds)


if __name__ == "__main__":
    main()
