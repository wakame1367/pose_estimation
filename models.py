from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.layers import Conv2D

from utils import keypoints2index

kp2index = keypoints2index()


def posenet(base_model_name="mobilenet"):
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

    kp_maps = Conv2D(len(kp2index),
                     kernel_size=(1, 1),
                     activation="sigmoid")(new_model.output)
    return [kp_maps]
