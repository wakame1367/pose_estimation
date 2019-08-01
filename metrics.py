import tensorflow.python.keras.backend as kb


# https://github.com/octiapp/KerasPersonLab/blob/32d44dd1f33377128a87d6e074cf8214224f0174/train.py#L57
def identity_metric(y_true, y_pred):
    return kb.mean(y_pred)
