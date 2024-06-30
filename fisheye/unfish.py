#! /usr/bin/env python3

from time import perf_counter

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
from numpy import ndarray
from supervision import FPSMonitor, Point, VideoInfo, draw_text
from typer import run
from vidgear.gears import VideoGear, WriteGear


class FisheyeFlatten:
  def __init__(
    self,
    reso: tuple[int, int],
    aspect_ratio: float | None = 1,
          camera_matrix = None,
          dist_coeffs = None
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
  camera_matrix = np.load('camera_matrix.npy')
  dist_coeffs = np.load('dist_coeffs.npy')
  if source.endswith('.mp4'):
    export_vid(source, out)
  elif source.endswith('.jpg'):
    img = imread(source)
    flatten = FisheyeFlatten(img.shape[:2][::-1], 1, camera_matrix, dist_coeffs)

    start = perf_counter()
    img = flatten(img)
    print(f'Taken: {perf_counter() - start}s')

    imwrite(f'{out}.jpg', (img))


def export_vid(source, out):
  mon = FPSMonitor()
  stream = VideoGear(source=source).start()
  writer = WriteGear(f'{out}.mp4')
  reso = VideoInfo.from_video_path(source).resolution_wh
  flatten = FisheyeFlatten(reso, 1)

  while (f := stream.read()) is not None:
    mon.tick()
    fps = mon()
    f = flatten(f)
    draw_text(
      scene=f,
      text=f'{fps:.1f}',
      text_anchor=Point(x=50, y=20),
      text_scale=1,
    )
    imshow('', f)
    if waitKey(1) & 0xFF == ord('q'):
      break
    writer.write(f)

  stream.stop()
  writer.close()
  destroyAllWindows()


if __name__ == '__main__':
  run(app)
