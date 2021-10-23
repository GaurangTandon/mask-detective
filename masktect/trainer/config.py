import tensorflow as tf

CFG = dict(
    net_count=7,
    batch_size=16,

    read_size=256,
    crop_size=250,
    net_size=224,

    LR_START=0.000005,
    LR_MAX=0.000020,
    LR_MIN=0.000001,
    LR_RAMPUP_EPOCHS=5,
    LR_SUSTAIN_EPOCHS=0,
    LR_EXP_DECAY=0.8,
    epochs=12,

    rot=180.0,
    shr=2.0,
    hzoom=8.0,
    wzoom=8.0,
    hshift=8.0,
    wshift=8.0,

    optimizer='adam',
    label_smooth_fac=0.05,

    tta_steps=25
)

AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = 1
