from tensorflow.python.keras.applications.mobilenet import MobileNet


def posenet(base_model_name="mobilenet"):
    base_model_names = ["mobilenet"]
    if base_model_name not in base_model_names:
        raise ValueError("{} only.".format(base_model_names[0]))

    if base_model_name == "mobilenet":
        base_model = MobileNet(include_top=False)
