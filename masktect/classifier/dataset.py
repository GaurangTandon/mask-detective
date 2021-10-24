import albumentations as albu

from ImageDataAugmentor.image_data_augmentor import ImageDataAugmentor

from .config import CONFIG


train_augment = albu.Compose(
    [
        albu.HorizontalFlip(p=0.5),
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
        albu.Resize(50, 50),
    ]
)

train_generator = ImageDataAugmentor(
    rescale=1.0 / 255, augment=train_augment, preprocess_input=None
).flow_from_directory(
    "data/mask/",
    target_size=(224, 224),
    batch_size=CONFIG["batch_size"],
    class_mode="binary",
)
