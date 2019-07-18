from pathlib import Path

import tensorflow as tf
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.layers import Conv2D, Lambda
from tensorflow.python.keras.models import Model

from utils import Config

config_path = Path("config.yaml")
config = Config(path=config_path)
kp2index = config.keypoints2index()
edges = config.edges()


def posenet(input_shape=(224, 224, 3), base_model_name="mobilenet"):
    base_model_names = ["mobilenet"]
    if base_model_name not in base_model_names:
        raise ValueError("{} only.".format(base_model_names[0]))

    if base_model_name == "mobilenet":
        base_model = MobileNet(input_shape=(224, 224, 3), include_top=False)
        # 14 * 14 * keypoints
        target_layer = "conv_pw_11_relu"
        new_model = Model(inputs=base_model.input,
                          outputs=base_model.get_layer(name=target_layer).output)
    else:
        raise ValueError()

    _kp_maps = Conv2D(len(kp2index), kernel_size=(1, 1), activation="sigmoid", name="heatmap")(new_model.output)
    _short_offsets = Conv2D(2 * len(kp2index), kernel_size=(1, 1), name="short_offset")(new_model.output)
    _mid_fwd_offsets = Conv2D(2 * len(edges), kernel_size=(1, 1), name="mid_fwd_offset")(new_model.output)
    _mid_bwd_offsets = Conv2D(2 * len(edges), kernel_size=(1, 1), name="mid_bwd_offset")(new_model.output)
    _segmentation_mask = Conv2D(1, kernel_size=(1, 1), activation="sigmoid", name="segmentation")(new_model.output)
    _long_offsets = Conv2D(2 * len(kp2index), kernel_size=(1, 1), name="long_offset")(new_model.output)

    image_shape = input_shape[:2]
    resize_layer = Lambda(lambda l: tf.image.resize_bilinear(l, image_shape, align_corners=True))
    _kp_maps = resize_layer(_kp_maps)
    _short_offsets = resize_layer(_short_offsets)
    _mid_bwd_offsets = resize_layer(_mid_bwd_offsets)
    _mid_fwd_offsets = resize_layer(_mid_fwd_offsets)
    _segmentation_mask = resize_layer(_segmentation_mask)
    _long_offsets = resize_layer(_long_offsets)

    return [_kp_maps, _short_offsets, _mid_fwd_offsets, _mid_bwd_offsets, _long_offsets, _segmentation_mask]
