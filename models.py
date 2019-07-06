from utils import keypoints2index
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.applications.mobilenet import MobileNet

kp2index = keypoints2index()


def posenet(base_model_name="mobilenet"):
    base_model_names = ["mobilenet"]
    if base_model_name not in base_model_names:
        raise ValueError("{} only.".format(base_model_names[0]))

    if base_model_name == "mobilenet":
        base_model = MobileNet(include_top=False)
    else:
        raise ValueError()

    kp_maps = Conv2D(len(kp2index),
                     kernel_size=(1, 1),
                     activation="sigmoid")(base_model.output)
    return [kp_maps]
