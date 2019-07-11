import json


def get_keypoints(keypoint_path):
    with open(keypoint_path, "r") as f:
        keypoints = json.load(f)

    image_info = {}  # key : image-id  value : keypoints
    for image in keypoints["images"]:
        image_info[image["id"]] = [image["file_name"]]

    # annotation
    for annotation in keypoints["annotations"]:
        image_id = annotation["image_id"]
        if image_id in image_info:
            image_info[image_id].append(annotation["keypoints"])

    return image_info