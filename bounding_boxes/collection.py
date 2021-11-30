from typing import Optional, List, Tuple, Set

import numpy as np
from sklearn.cluster import KMeans

from PIL import Image, ImageDraw

from bounding_boxes.utils import Colors


def _cluster_corner_points(
    bounding_boxes: np.ndarray, n_clusters: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param bounding_boxes: Numpy array of shape (N, 2, 2) where N is the number of points. The entries
        `bounding_boxes[i, 0]` refer to the coordinates of the upper left and `bounding_boxes[i, 1]` to the
        coordinates of the lower right point of the i-th bounding box.
    :n_clusters: An integer number of clusters.
    :return: A pair of numpy arrays of shape (N,) each. The first array denotes the clustering of the upper left
        corners of the bounding boxes while the second array is the clustering of the lower right corners of the
        bounding boxes.
    """
    upper_left = bounding_boxes[:, 0]
    lower_right = bounding_boxes[:, 1]
    kmeans_ul = KMeans(n_clusters=n_clusters).fit(upper_left)
    kmeans_lr = KMeans(n_clusters=n_clusters).fit(lower_right)
    assignments_ul = kmeans_ul.labels_
    assignments_lr = kmeans_lr.labels_
    return assignments_ul, assignments_lr


def _find_partition(
    assignments_ul: np.ndarray, assignments_lr: np.ndarray
) -> List[Set]:
    """
    Based on the clustering results of the upper left and lower right corners of the bounding boxes, we determine the
        equivalence classes, i.e., the sets of instances for which the upper left and lower right corners each lie in
        the same cluster.

    Note: if the assignments come from a clustering with k clusters, then the resulting partition has at least, but not
        necessarily exactly, k classes.
    :param assignments_ul: (N,) numpy array of cluster assignments of the upper left points.
    :param assignments_lr: (N,) numpy array of cluster assignments of the lower right points.
    :return: A list of sets, where each set contains integer indices representing the bounding boxes.
    """
    num_annotators = len(assignments_ul)
    partition = list()

    def is_equivalent(i: int, j: int) -> bool:
        return all(
            [
                assignments_ul[i] == assignments_ul[j],
                assignments_lr[i] == assignments_lr[j],
            ]
        )

    for r in range(num_annotators):
        already_exists = any([r in C for C in partition])
        # If the chosen box is not already located in one of the classes
        if not already_exists:
            # Create a new equivalence class for the representative r
            C = set()
            for s in range(num_annotators):
                if is_equivalent(r, s):
                    C.add(s)
            partition.append(C)

    return partition


class BoundingBoxCollection:
    def __init__(self, upper_left: np.ndarray, height: np.ndarray, width: np.ndarray):
        """
        A bounding box collection is created by specifying the individual bounding boxes. Each bounding box is
            identified by the upper left corner, as well as the height and width of the box. Internally, the box is
            represented by the upper left corner and the lower right corner.

        :param upper_left: (N, 2) numpy array containing the upper left point of each box.
        :param height: (N,) numpy array containing the height of each box.
        :param width: (N,) numpy array containing the width of each box.
        """
        self.bounding_boxes = BoundingBoxCollection._create_array(
            upper_left, height, width
        )

    @staticmethod
    def _create_array(
        upper_left: np.ndarray, height: np.ndarray, width: np.ndarray
    ) -> np.ndarray:
        """
        :param upper_left: (N, 2) numpy array containing the upper left point of each box.
        :param height: (N,) numpy array containing the height of each box.
        :param width: (N,) numpy array containing the width of each box.

        :return:
            A (N, 2, 2) numpy array storing the information of the bounding boxes as collections of upper left and
                lower right corner points.
        """

        offset = np.concatenate(
            [np.expand_dims(width, axis=1), np.expand_dims(height, axis=1)], axis=1
        )

        ul = np.expand_dims(upper_left, axis=1)
        lr = np.expand_dims(upper_left + offset, axis=1)

        return np.concatenate([ul, lr], axis=1)

    def get_dummy_partition(self) -> List[Set]:
        """
        Returns a dummy partition or the default partition where each bounding box is assigned to its own class.
        :return: A list containing one-element sets of indices of bounding boxes.
        """
        num_boxes = len(self.bounding_boxes)
        partition = [{idx} for idx in range(num_boxes)]
        return partition

    def get_transformed_boxes(self) -> np.ndarray:
        """
        Flattens the bounding boxes, i.e. transforms the data in an array of shape (N, 4)
        :return: A numpy array of shape (N, 4) containing the coordinates of the upper left and lower right corner
            points of the bounding boxes.
        """
        num_bounding_boxes, *tail = self.bounding_boxes.shape
        return self.bounding_boxes.reshape((num_bounding_boxes, -1))

    def get_size(self) -> Tuple[int, int]:
        """
        This is only an auxiliary function to calculate the size of the resulting virtual image.
        :return: A pair of integers specifying the extent of the image.
        """
        lower_right = self.bounding_boxes[:, 1]  # (N, 2)
        max_point = np.max(lower_right, axis=0)
        # We 'pad' the image a little bit, i.e. we consider 120% of the maximal extent
        max_x, max_y = (1.2 * max_point).astype(int)
        return max_x, max_y

    def create_image(
        self,
        partition: Optional[Tuple[Set]] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """
        Creates a PIL image that shows all the bounding boxes of the collection on a black background.

        :param partition: The respective partition of the set of bounding boxes, i.e. a list of sets. The partition is
            responsible for the color in which the bounding boxes are displayed. Each two instances of the same group
            are displayed in the same color.
        :param size: The size of the resulting image. If no size is set, the size is determined automatically according
            to the maximum extent of the boxes.
        :return: The resulting PIL image with all of the bounding boxes drawn on a black background.
        """
        im = Image.new(mode="RGB", size=size if size is not None else self.get_size())
        draw = ImageDraw.Draw(im)

        transformed_boxes = self.get_transformed_boxes()
        num_boxes = len(transformed_boxes)

        # If no partition is set, use the trivial one
        if partition is None:
            partition = self.get_dummy_partition()

        colors = ["black"] * num_boxes
        for class_idx, C in enumerate(partition):
            for bbox_idx in C:
                colors[bbox_idx] = Colors.get(class_idx)

        for idx, bbox in enumerate(transformed_boxes):
            xy = bbox.tolist()
            draw.rectangle(xy, outline=colors[idx])

        return im

    def show(self, partition: Optional[Tuple[Set]] = None, **kwargs):
        im = self.create_image(partition=partition, **kwargs)
        im.show()

    def save_image(self, partition: Optional[Tuple[Set]] = None, **kwargs):
        im = self.create_image(partition=partition, **kwargs)

        def _save(filename: str):
            im.save(filename)

        return _save

    def find_partition(self, n_clusters: int) -> List[Set]:
        """
        Computes a partition of the bounding boxes, based on a clustering with `n_clusters` many clusters.
        :param n_clusters: An integer specifying the number of clusters.
        :return: A partition, i.e. a list of sets of integers.
        """
        assignment = _cluster_corner_points(self.bounding_boxes, n_clusters)
        partition = _find_partition(*assignment)
        return partition

    def compute_pairwise_iou(self) -> np.ndarray:
        """
        Calculates the IoU for each pair of bounding boxes and writes them into a matrix.
        :return: An (N, N) numpy array, where the entry i, j contains the IoU between the i-th and j-th bounding boxes.
        """

        # First identify the coordinates of the intersecting boxes
        bbox1 = np.expand_dims(self.bounding_boxes, axis=1)
        bbox2 = np.expand_dims(self.bounding_boxes, axis=0)
        _max = np.maximum(bbox1, bbox2)
        _min = np.minimum(bbox1, bbox2)
        upper_left = _max[:, :, 0]
        lower_right = _min[:, :, 1]

        # Next, we need to consider if two boxes intersect at all
        points_ul = bbox1[:, :, 0]
        points_lr = bbox2[:, :, 1]

        # A criterion for when two boxes can be considered intersecting
        criterion = np.all(points_ul <= points_lr, axis=-1)
        is_intersecting = np.logical_and(criterion, criterion.T)

        intersection_bounding_boxes = np.concatenate(
            [np.expand_dims(upper_left, axis=2), np.expand_dims(lower_right, axis=2)],
            axis=2,
        )

        # For each intersecting pair, compute IoU
        # If no intersection => set IoU to zero

        intersection_bounding_boxes[~is_intersecting] = 0

        # Compute volume (i.e. area) of intersecting boxes
        upper_intersect = intersection_bounding_boxes[:, :, 0]
        lower_intersect = intersection_bounding_boxes[:, :, 1]
        intersection_volume = np.prod(lower_intersect - upper_intersect, axis=-1)

        # Compute total volume of the boxes
        upper = self.bounding_boxes[:, 0]
        lower = self.bounding_boxes[:, 1]
        volume = np.prod(lower - upper, axis=-1)

        union_volume = (
            np.expand_dims(volume, axis=1)
            + np.expand_dims(volume, axis=0)
            - intersection_volume
        )

        iou = intersection_volume / union_volume

        return iou

    def compute_mean_weighted_iou(
        self, partition: Optional[Tuple[Set]] = None
    ) -> float:
        """
        Computes for a given partition the mean (weighted) IoU.
        :param partition: None or a partition, i.e. a list of sets of integer indices.
        :return: A score value that ranks the give partition.
        """
        if partition is None:
            partition = self.get_dummy_partition()

        iou = self.compute_pairwise_iou()

        weighted_group_ious = list()

        for C in partition:
            # Project IoU matrix onto set C
            indices = np.array(list(C))
            g = len(C)

            iou_projected = iou[indices.reshape(-1, 1), indices.reshape(1, -1)]

            total_sum = np.sum(iou_projected)

            # Note g is the number of elements in the group
            # But g is also the trace of the matrix `iou_projected`

            gamma = 1 / (g + 3)

            def _weighted_iou(sigma):
                return (1 - gamma) / (g * (g + 1)) * (sigma - g) + gamma

            weighted_iou = _weighted_iou(total_sum)

            weighted_group_ious.append(weighted_iou)

        return np.array(weighted_group_ious).mean()

    def compute_iou_from_clustering(
        self, n_clusters_max: int = 5, repeats_per_clustering: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For different numbers of possible clusters, we use kmeans clustering to cluster the upper left and lower right
            corners of all bounding boxes. We then form a partition of the set of all bounding boxes based on whether
            for every two bounding boxes the respective vertices landed in the same cluster.

        Since we do not know the number of actual bounding boxes, we consider different possible numbers of clusters.

        :param n_clusters_max: The maximum number of clusters to consider.
        :type n_clusters_max:
        :param repeats_per_clustering: How often should we repeat the clustering for each number of clusters?
        :return: A pair of numpy arrays of length `n_clusters_max`. The two arrays represent the mean weighted IoU, and
            the standard deviation resulting from the repetitions per number of clusters.
        """
        mean, std = list(), list()
        for n_clusters in range(1, n_clusters_max + 1):
            iou_from_clustering = list()
            for _ in range(repeats_per_clustering):
                # Compute the partition for a given number of clusters and the score of this assignment
                partition = self.find_partition(n_clusters=n_clusters)
                mean_weighted_iou = self.compute_mean_weighted_iou(partition)
                iou_from_clustering.append(mean_weighted_iou)
            iou_from_clustering = np.array(iou_from_clustering)
            mean.append(iou_from_clustering.mean())
            std.append(iou_from_clustering.std())
        return mean, std

    def find_best_partition(self, **kwargs) -> List[Set]:
        """
        For different numbers of clusters, we form the partitions of the set of bounding boxes and then compute the
            goodness of these aggregations. Finally, we select the partition with the highest score. Note that the
            number of clusters is only a lower bound on the number of resulting aggregated boxes (since the corners
            must be clustered separately and not lead to any agreement).

        :return: The optimal partition of the set of bounding boxes.
        """
        iou, _ = self.compute_iou_from_clustering(**kwargs)

        # Note: clustering with k means that the resulting number of bounding boxes is AT LEAST k.
        n_clusters_best = 1 + np.argmax(iou)
        partition = self.find_partition(n_clusters=n_clusters_best)

        return partition

    def aggregate(self, **kwargs):
        """
        Calculates the median bounding box within the groups of the best partition.
        :return: A bounding box collection of median bounding boxes.
        """
        partition = self.find_best_partition(**kwargs)
        upper_left_total = list()
        height_total = list()
        width_total = list()

        # Aggregate within each group

        for C in partition:
            indices = np.array(list(C))
            selected_boxes = self.bounding_boxes[indices]
            upper_left = np.mean(selected_boxes[:, 0], keepdims=True, axis=0)
            lower_right = np.mean(selected_boxes[:, 1], keepdims=True, axis=0)
            box_size = lower_right - upper_left
            width, height = box_size[:, 0], box_size[:, 1]
            upper_left_total.append(upper_left)
            height_total.append(height)
            width_total.append(width)

        upper_left_total = np.concatenate(upper_left_total, axis=0)
        height_total = np.concatenate(height_total, axis=0)
        width_total = np.concatenate(width_total, axis=0)

        return BoundingBoxCollection(upper_left_total, height_total, width_total), partition
