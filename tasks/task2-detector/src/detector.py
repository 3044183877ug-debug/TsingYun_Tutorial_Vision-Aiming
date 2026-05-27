"""Task 2 MNIST-board detector helpers with student TODO extension points.

This file belongs to Task 2. The simulator runner imports it so that a Task 2
implementation can be tested both offline and inside the Unity simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from simulator_client.protocol import Matrix3x3
from model import classify_mnist_digit

Point2D = tuple[float, float]
CornerSet = tuple[Point2D, Point2D, Point2D, Point2D]
RgbPixel = tuple[int, int, int]
ImageLike = np.ndarray
WARP_OUTPUT_SIZE = 128
MNIST_INNER_RATIO = 0.69


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Point2D:
        return (self.x + self.width * 0.5, self.y + self.height * 0.5)


@dataclass
class Detection:
    class_id: int
    confidence: float
    bbox: BoundingBox
    corners: CornerSet
    rvec: object | None = None
    tvec: object | None = None


def _bbox_from_corners(corners: Sequence[Point2D]) -> BoundingBox:
    if len(corners) != 4:
        raise ValueError(f"Expected 4 corners, got {len(corners)}")

    xs = [float(point[0]) for point in corners]
    ys = [float(point[1]) for point in corners]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    return BoundingBox(
        x=min_x,
        y=min_y,
        width=max_x - min_x + 1.0,
        height=max_y - min_y + 1.0,
    )


def _crop_bounds(corners: Sequence[Point2D], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    bbox = _bbox_from_corners(corners)
    x0 = max(0, min(image_width, int(np.floor(bbox.x))))
    y0 = max(0, min(image_height, int(np.floor(bbox.y))))
    x1 = max(0, min(image_width, int(np.ceil(bbox.x + bbox.width))))
    y1 = max(0, min(image_height, int(np.ceil(bbox.y + bbox.height))))
    return x0, y0, x1, y1


def crop_bbox(image: np.ndarray, corner_candidates: Sequence[Sequence[Point2D]]) -> list[np.ndarray]:
    crops: list[np.ndarray] = []
    for corners in corner_candidates:
        if len(corners) != 4:
            continue

        # `corners` are expected in LU, RU, RD, LD order.
        src = np.array(corners, dtype=np.float32)

        # Shrink the source quad toward its center so the warp removes the outer red border.
        center = np.mean(src, axis=0)
        src = center + (src - center) * MNIST_INNER_RATIO

        dst = np.array(
            [
                [0, 0],
                [WARP_OUTPUT_SIZE - 1, 0],
                [WARP_OUTPUT_SIZE - 1, WARP_OUTPUT_SIZE - 1],
                [0, WARP_OUTPUT_SIZE - 1],
            ],
            dtype=np.float32,
        )

        perspective = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            image,
            perspective,
            (WARP_OUTPUT_SIZE, WARP_OUTPUT_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        crops.append(warped)
    return crops


def order_corners(corners: Sequence[Point2D]) -> CornerSet:
    # TODO(student): Sort the four target corners into a stable order.
    # Input: four 2D corners in arbitrary order.
    # Output: corners ordered as top-left, top-right, bottom-right, bottom-left.
    # Compute a stable ordering rule that works for the target board geometry.
    pts = np.array(corners, dtype=np.float32)
    x_sorted = pts[np.argsort(pts[:, 0])]
    left_pts = x_sorted[:2]
    right_pts = x_sorted[2:]
    left_pts = left_pts[np.argsort(left_pts[:, 1])]
    lu = (float(left_pts[0][0]), float(left_pts[0][1]))
    ld = (float(left_pts[1][0]), float(left_pts[1][1]))
    right_pts = right_pts[np.argsort(right_pts[:, 1])]
    ru = (float(right_pts[0][0]), float(right_pts[0][1]))
    rd = (float(right_pts[1][0]), float(right_pts[1][1]))
    return (lu, ru, rd, ld)



def detect_bbox(image: ImageLike, threshold: int = 200) -> list[CornerSet]:
    # TODO(student): Detect board candidates.
    # image_array = convert image to an OpenCV-compatible uint8 array
    # red_mask = threshold reddish pixels into a binary image
    # optionally clean red_mask with morphology so small noisy blobs disappear
    # contours = cv2.findContours(red_mask)
    # corner_candidates = []
    # for each contour:
    #     if contour area is too small:
    #         continue
    #     polygon = cv2.approxPolyDP(contour, epsilon, closed=true)
    #     if polygon does not have exactly 4 edges/corners:
    #         continue
    #     if polygon is not convex or has unreasonable aspect ratio:
    #         continue
    #     corners = order_corners(the four polygon vertices)
    #     append corners to corner_candidates
    # return corner_candidates
    image_array = np.array(image, dtype=np.uint8)
    r_channel = image_array[:, :, 0].astype(int)
    g_channel = image_array[:, :, 1].astype(int)
    b_channel = image_array[:, :, 2].astype(int)  
    mask = (r_channel > threshold) & (r_channel > g_channel + 20) & (r_channel > b_channel + 20)
    red_mask = (mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corner_candidates = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
        epsilon = 0.04 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(polygon) != 4:
            continue
        if not cv2.isContourConvex(polygon):
            continue
        x, y, w, h = cv2.boundingRect(polygon)
        if w == 0 or h == 0 or not (0.2 < w / float(h) < 5.0):
            continue
        raw_corners = [(float(pt[0][0]), float(pt[0][1])) for pt in polygon]
        ordered_corners = order_corners(raw_corners)
        corner_candidates.append(ordered_corners)
    return corner_candidates



def detect_mnist_board(image: ImageLike, threshold: int = 200) -> list[Detection]:
    # TODO(student): Classify detected MNIST-board candidates and filter them.
    # Input: one RGB image and a threshold parameter.
    # Output: a list of Detection objects.
    # Step 1: call detect_bbox(...) to get board candidates.
    # Step 2: call crop_bbox(...) to extract candidate crops.
    # Step 3: call classify_mnist_digit(...) on each crop and filter low-confidence results.
    # Step 4: package the remaining results as Detection objects.
    candidates = detect_bbox(image, threshold)
    if not candidates:
        return []
    crops = crop_bbox(image, candidates)
    detections = []
    for corners, crop in zip(candidates, crops):
        digit, confidence = classify_mnist_digit(crop)
        if confidence >= 0.5 and digit != 0:
            bbox = _bbox_from_corners(corners)
            detections.append(Detection(
                class_id=digit,
                confidence=confidence,
                bbox=bbox,
                corners=corners
            ))
    return detections


def solve_pnp(

    # TODO(student): Fill rvec and tvec for every valid Detection.
    # half_width = board_width_meters / 2
    # half_height = board_height_meters / 2
    # object_points = four physical board corners as float32:
    #     (-half_width, -half_height, 0)
    #     (half_width, -half_height, 0)
    #     (half_width, half_height, 0)
    #     (-half_width, half_height, 0)
    # camera_array = camera_matrix as a 3x3 float64 array
    # dist_array = zero distortion if dist_coeffs is not provided
    # result = []
    # for each detection:
    #     image_points = detection.corners as a float32 4x2 array
    #     call cv2.solvePnP with object_points, image_points, camera_array, and dist_array
    #     if OpenCV reports failure:
    #         skip this detection or raise a clear error
    #     fill detection.rvec and detection.tvec with the OpenCV result
    #     append detection to result
    # return result
    detections: Sequence[Detection],
    camera_matrix: Matrix3x3,
    board_width_meters: float,
    board_height_meters: float,
    dist_coeffs: Sequence[float] | None = None,
) -> list[Detection]:
    half_width = board_width_meters / 2.0
    half_height = board_height_meters / 2.0
    
    object_points = np.array([
        [-half_width, -half_height, 0.0],
        [half_width, -half_height, 0.0],
        [half_width, half_height, 0.0],
        [-half_width, half_height, 0.0]
    ], dtype=np.float32)
    
    camera_array = np.array(camera_matrix, dtype=np.float64)
    
    if dist_coeffs is None:
        dist_array = np.zeros((4, 1), dtype=np.float64)
    else:
        dist_array = np.array(dist_coeffs, dtype=np.float64)
        
    result = []
    for detection in detections:
        image_points = np.array(detection.corners, dtype=np.float32)
        
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_array, dist_array
        )
        
        if success:
            detection.rvec = rvec
            detection.tvec = tvec
            result.append(detection)
            
    return result