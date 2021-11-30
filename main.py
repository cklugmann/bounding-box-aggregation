import os

import numpy as np
from bounding_boxes.collection import BoundingBoxCollection


def main(out_dir: str = "out"):
    """

    This is just one example that shows how to group and aggregate bounding boxes.

    """
    upper_left = np.array(
        [
            [32.0, 16.0],
            [30, 20],
            [64, 16],
            [68, 20],
            [60, 40],
            [128, 256],
            [116, 280],
            [100, 98],
            [80, 60],
            [85, 58],
            [65, 32],
        ]
    )
    height = np.array([32, 40, 84, 80, 60, 80, 74, 32, 256, 210, 40])
    width = np.array([16, 14, 32, 26, 32, 80, 70, 60, 128, 90, 35])
    boundig_boxes = BoundingBoxCollection(upper_left, height, width)

    os.makedirs(out_dir, exist_ok=True)

    def save_name(name: str, filetype: str = "png") -> str:
        return os.path.join(out_dir, "{}.{}".format(name, filetype))

    boundig_boxes.save_image()(save_name("raw_boxes"))

    aggregated_bounding_boxes, best_partition = boundig_boxes.aggregate()

    boundig_boxes.save_image(partition=best_partition)(save_name("grouped_boxes"))

    size = boundig_boxes.get_size()
    aggregated_bounding_boxes.save_image(size=size)(save_name("aggregated_boxes"))


if __name__ == "__main__":
    main()
