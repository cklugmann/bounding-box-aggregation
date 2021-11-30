from typing import List


class Colors:

    # This is a list of colors that the bounding boxes are assigned to. Once the list is exhausted, it starts
    #   again with the first entry.
    AVAILABLE_COLORS: List[str] = ["red", "blue", "green", "yellow", "orange"]

    def __init__(self):
        pass

    @staticmethod
    def num_available_colors():
        return len(Colors.AVAILABLE_COLORS)

    @staticmethod
    def get(idx: int) -> str:
        return Colors.AVAILABLE_COLORS[idx % Colors.num_available_colors()]
