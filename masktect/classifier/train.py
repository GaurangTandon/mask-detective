import tensorflow as tf

from .config import CONFIG
from .dataset import train_generator


def build_model():
    model_input = tf.keras.Input(
        shape=(CONFIG["net_size"], CONFIG["net_size"], 3), name="image_input"
    )
    x = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights="imagenet",
        input_shape=(CONFIG["net_size"], CONFIG["net_size"], 3),
    )(model_input)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(model_input, x, name="network")
    model.summary()

    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(
        optimizer=CONFIG["optimizer"],
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def train_model():
    network = build_model()
    callbacks_list = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="train_loss", patience=2, factor=0.1, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "model_checkpoint",
            monitor="loss",
            save_weights_only=True,
            save_best_only=True,
        ),
    ]
    history = network.fit(
        train_generator,
        verbose=1,
        steps_per_epoch=len(train_generator),
        epochs=CONFIG["epochs"],
        callbacks=callbacks_list,
    )
    network.save("weights/model.h5")
    return history


if __name__ == "__main__":
    train_model()
