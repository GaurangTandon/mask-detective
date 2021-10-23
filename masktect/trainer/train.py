import efficientnet.tfkeras as efn
import tensorflow as tf
from .config import CFG

REPLICAS = 1

# default CPU/GPU strategy
strategy = tf.distribute.get_strategy()


def get_lr_callback(cfg):
    lr_start = cfg['LR_START']
    lr_max = cfg['LR_MAX'] * strategy.num_replicas_in_sync
    lr_min = cfg['LR_MIN']
    lr_ramp_ep = cfg['LR_RAMPUP_EPOCHS']
    lr_sus_ep = cfg['LR_SUSTAIN_EPOCHS']
    lr_decay = cfg['LR_EXP_DECAY']

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max

        else:
            lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min

        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback


def get_model(cfg):
    model_input = tf.keras.Input(shape=(cfg['net_size'], cfg['net_size'], 3), name='imgIn')

    dummy = tf.keras.layers.Lambda(lambda x: x)(model_input)

    outputs = []
    for i in range(cfg['net_count']):
        constructor = getattr(efn, f'EfficientNetB{i}')

        x = constructor(include_top=False, weights='imagenet',
                        input_shape=(cfg['net_size'], cfg['net_size'], 3),
                        pooling='avg')(dummy)

        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        outputs.append(x)

    model = tf.keras.Model(model_input, outputs, name='aNetwork')
    model.summary()
    return model


def compile_new_model(cfg):
    with strategy.scope():
        model = get_model(cfg)

        losses = [tf.keras.losses.BinaryCrossentropy(label_smoothing=cfg['label_smooth_fac'])
                  for i in range(cfg['net_count'])]

        model.compile(
            optimizer=cfg['optimizer'],
            loss=losses,
            metrics=[tf.keras.metrics.Accuracy(name='accuracy')])

    return model


def train_model(save_path):
    steps_train = (23871 + 29852) / (CFG['batch_size'] * REPLICAS)
    steps_val = (2651 + 3316) / (CFG['batch_size'] * REPLICAS)

    model = compile_new_model(CFG)
    history = model.fit(ds_train,
                        verbose=1,
                        steps_per_epoch=steps_train,
                        epochs=CFG['epochs'],
                        callbacks=[get_lr_callback(CFG)],
                        validation_data=ds_val,
                        validation_steps=steps_val
                        )

    model.save(save_path)