import yaml


class Config:
    def __init__(self, path):
        with open(path, "r") as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        self.configs = configs

    def keypoints2index(self):
        """

        :return:
        """
        keypoints = self.configs["keypoints"]
        return dict(zip(range(len(keypoints)), keypoints))

    def edges(self):
        return self.configs["edges"]

    def right_and_left_parts_indexes(self):
        """

        :return: list, list
        """
        right_indexes = []
        left_indexes = []
        for part_idx, part_name in self.keypoints2index().items():
            if part_name.startswith("R"):
                right_indexes.append(part_idx)
            elif part_name.startswith("L"):
                left_indexes.append(part_idx)
        return right_indexes, left_indexes
