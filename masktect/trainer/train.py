import efficientnet.tfkeras as efn
import tensorflow as tf
from .config import CFG
from .dataset import train_gen

REPLICAS = 1

# default CPU/GPU strategy
strategy = tf.distribute.get_strategy()

def get_model(cfg):
    model_input = tf.keras.Input(shape=(cfg['net_size'], cfg['net_size'], 3), name='imgIn')

    outputs = []
    # for i in range(4, 5):
    # constructor = getattr()
    x = tf.keras.applications.EfficientNetB2(include_top=False, weights='imagenet',
                    input_shape=(cfg['net_size'], cfg['net_size'], 3))(model_input)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    outputs.append(x)

    model = tf.keras.Model(model_input, outputs, name='aNetwork')
    model.summary()
    return model


def compile_new_model(cfg):
    with strategy.scope():
        model = get_model(cfg)

        loss = tf.keras.losses.BinaryCrossentropy()
        model.compile(
            optimizer=cfg['optimizer'],
            loss=loss,
            metrics=[tf.keras.metrics.Accuracy(name='accuracy')])

    return model


def train_model(save_path):
    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='train_loss', patience=2, factor=0.1, verbose=1)
    chkpnt = tf.keras.callbacks.ModelCheckpoint('check_effnet', monitor='train_loss',
                                                save_weights_only=True, save_best_only=True)
    model = compile_new_model(CFG)
    history = model.fit(train_gen,
                        verbose=1,
                        steps_per_epoch=len(train_gen),
                        epochs=CFG['epochs'],
                        callbacks=[lr, chkpnt],
                        )
    model.save(save_path)

if __name__ == "__main__":
    train_model("model.h5")