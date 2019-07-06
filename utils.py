import yaml


def keypoints2index():
    """

    :return:
    """
    with open("config.yaml") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    keypoints = configs["keypoints"]
    return dict(zip(range(len(keypoints)), keypoints))


def right_and_left_parts_indexes(ky2index):
    """

    :param ky2index: dict
    :return: list, list
    """
    right_indexes = []
    left_indexes = []
    for part_idx, part_name in ky2index.items():
        if part_name.startswith("R"):
            right_indexes.append(part_idx)
        elif part_name.startswith("L"):
            left_indexes.append(part_idx)
    return right_indexes, left_indexes
