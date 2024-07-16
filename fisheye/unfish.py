import os

import numpy as np
from cv2 import (
    destroyAllWindows,
    getOptimalNewCameraMatrix,
    imread,
    imshow,
    imwrite,
    undistort,
    waitKey,
)
import cv2
from numpy import ndarray
from supervision import VideoInfo
from typer import run

camera_matrix = np.load('F:/graduate_thesis/checkpoints/fisheye/camera_matrix.npy')
dist_coeffs = np.load('F:/graduate_thesis/checkpoints/fisheye/dist_coeffs.npy')


class FisheyeFlatten:
    def __init__(
            self,
            reso: tuple[int, int],
            aspect_ratio: float | None = 1,
            camera_matrix=None,
            dist_coeffs=None,
    ):
        w, h = reso
        s = min(w, h)
        d = abs(w - h) // 2
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.new_camera_matrix, roi = getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (s, s), 1, (s, s)
        )
        left, top, new_w, new_h = roi
        bottom = top + new_h
        right = left + new_w

        if aspect_ratio:
            if new_w / new_h > aspect_ratio:
                spare = (new_w - int(new_h * aspect_ratio)) // 2
                left += spare
                right -= spare
            else:
                spare = (new_h - int(new_w / aspect_ratio)) // 2
                top += spare
                bottom -= spare

        possible1pix = (right - left) - (bottom - top)
        if possible1pix > 0:
            bottom += 1
        elif possible1pix < 0:
            right += 1

        self.crop = slice(top, bottom), slice(left, right)
        self.slic = (
            (slice(None), slice(None))
            if w == h
            else ((slice(None), slice(d, d + h)) if w > h else (slice(d, d + w), slice(None)))
        )

    def __call__(self, f: ndarray) -> ndarray:
        return undistort(
            f[self.slic],
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.new_camera_matrix,
        )[self.crop]


def app(source: str, out: str = 'out'):
    image_path = os.path.join(source)
    export_vid(image_path, out)


def export_vid(source, out):
    width = 0
    height = 0
    i = 0
    vid_writer = None

    vid = cv2.VideoCapture(source)

    output_name = source.rsplit('/', 1)[-1]

    reso = VideoInfo.from_video_path(source).resolution_wh
    flatten = FisheyeFlatten(reso, 1, camera_matrix, dist_coeffs)

    while True:
        _, frame = vid.read()
        i += 1
        if frame is None:
            break
        f = flatten(frame)
        if (width == 0) or (height == 0):
            width = f.shape[1]
            height = f.shape[0]
            vid_writer = cv2.VideoWriter(f"F:/graduate_thesis/results/fisheye/{output_name}",
                                         cv2.VideoWriter_fourcc(*"mp4v"), 25, (width, height))

        vid_writer.write(f)
        cv2.imshow("Undisorted frame", f)
        cv2.waitKey(1)

    vid_writer.release()
    destroyAllWindows()


if __name__ == '__main__':
    run(app)
