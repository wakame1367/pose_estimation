import json


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
            image_info[image_id].append(annotation["keypoints"])

    return image_info