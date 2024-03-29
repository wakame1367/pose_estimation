from tensorflow.python.keras import backend as kb


def offset_loss(y_true, y_pred):
    return kb.mean(kb.abs((y_pred - y_true)), axis=-1)


def euclidean_distance_loss(y_true, y_pred):
    return kb.sqrt(kb.sum(kb.square((y_pred - y_true)), axis=-1))


def heatmap_loss(y_true, y_pred, coeff=4):
    return kb.mean(kb.binary_crossentropy(y_true, y_pred), axis=-1, keepdims=True) * coeff
