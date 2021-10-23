import albumentations as albu
import os
import PIL
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from ImageDataAugmentor.image_data_augmentor import *
from .config import CFG, REPLICAS, AUTO

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
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

train_augment = get_training_augmentation()

def prepare_image(img, cfg=None, is_test=False):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [cfg['read_size'], cfg['read_size']])
    img = tf.cast(img, tf.float32) / 255.0

    if not is_test:
        img = train_augment(image=img)
    # else:
        # img = val_augment(image=img)

    img = tf.image.resize(img, [cfg['net_size'], cfg['net_size']])
    img = tf.reshape(img, [cfg['net_size'], cfg['net_size'], 3])
    return img

def get_train_dataset(train_files_path, cfg):
    train_datagen = ImageDataAugmentor(
        rescale=1. / 255,
        augment=train_augment,
        preprocess_input=None)
    train_generator = train_datagen.flow_from_directory(
        train_files_path,
        target_size=(224, 224),
        batch_size=cfg['batch_size'],
        class_mode='binary')

    return train_generator

def get_val_dataset(val_files_path, cfg):
    val_datagen = ImageDataAugmentor(
        rescale=1. / 255)
    validation_generator = val_datagen.flow_from_directory(
        val_files_path,
        target_size=(224, 224),
        batch_size=cfg['batch_size'],
        class_mode='binary')

    return validation_generator

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
# WITH = "with_mask"
# WITHOUT = "without_mask"
# with_mask = os.listdir(f"{DATASET_PATH}/{WITH}")
# without_mask = os.listdir(f"{DATASET_PATH}/{WITHOUT}")
# list of (path, label)
# all_files_with_labels = [(f"{DATASET_PATH}/{WITH}/{x}", 1) for x in with_mask] + \
#                             [(f"{DATASET_PATH}/{WITHOUT}/{x}", 0) for x in without_mask]
# np.random.shuffle(all_files_with_labels)
# TRAIN_RATIO = 0.8
# TRAIN_COUNT = int(0.8 * len(all_files_with_labels))
# list of tuples (filepath, label)
# files_train = all_files_with_labels[:TRAIN_COUNT]
# list of filepaths
# files_test = [x for x, _ in all_files_with_labels[TRAIN_COUNT:]]

train_gen = get_train_dataset(DATASET_PATH, CFG)

def main():
    train_gen.show_data()

if __name__ == "__main__":
    main()
