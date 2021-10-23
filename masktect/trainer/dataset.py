import albumentations as albu
import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = 1


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

