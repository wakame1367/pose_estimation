from pathlib import Path

from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.layers import Conv2D, Input
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
        # new_model = Model(inputs=base_model.input,
        #                   outputs=base_model.get_layer(name=target_layer).output)
        out = base_model.get_layer(name=target_layer).output
    else:
        raise ValueError()

    _kp_maps = Conv2D(len(kp2index), kernel_size=(1, 1), activation="sigmoid", name="heatmap")(out)
    _short_offsets = Conv2D(2 * len(kp2index), kernel_size=(1, 1), name="offset")(out)

    return Model(inputs=base_model.input, outputs=[_kp_maps, _short_offsets])
