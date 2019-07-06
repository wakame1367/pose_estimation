import yaml


def keypoints2index():
    """

    :return:
    """
    with open("config.yaml") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    keypoints = configs["keypoints"]
    return dict(zip(range(len(keypoints)), keypoints))
