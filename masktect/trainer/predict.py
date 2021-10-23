from tensorflow import keras
from .config import CFG, REPLICAS, AUTO
import numpy as np


def predictor(model_path):
    model = keras.models.load_model(model_path)
    CFG['batch_size'] = 256

    cnt_test = 14927
    steps = cnt_test / (CFG['batch_size'] * REPLICAS) * CFG['tta_steps']
    ds_testAug = get_dataset(files_test, CFG, augment=True, repeat=True,
                             labeled=False, return_image_names=False)

    probs = model.predict(ds_testAug, verbose=1, steps=steps)

    probs = np.stack(probs)
    probs = probs[:, :cnt_test * CFG['tta_steps']]
    probs = np.stack(np.split(probs, CFG['tta_steps'], axis=1), axis=1)
    probs = np.mean(probs, axis=1)
    return probs
