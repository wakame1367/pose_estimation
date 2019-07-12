import json
import numpy as np


def get_keypoints(keypoint_path):
    """

    :param keypoint_path: pathlib.Path or str
    :return: dict
    """
    with open(keypoint_path, "r") as f:
        keypoints = json.load(f)

    image_info = {}  # key : image-id  value : keypoints
    for image in keypoints["images"]:
        image_info[image["id"]] = [image["file_name"]]

    # annotation
    for annotation in keypoints["annotations"]:
        image_id = annotation["image_id"]
        if image_info.get(image_id):
            print(len(annotation["keypoints"]))
            image_info[image_id].append(np.array(annotation["keypoints"]))
        else:
            image_info[image_id].append(None)
    """
    http://cocodataset.org/#format-data
    x, y, v
    x and y: location
    v: visibility flag
       - v=0: not labeled (in which case x=y=0)
       - v=1: labeled but not visible
       - v=2: labeled and visible
    """

    return image_info