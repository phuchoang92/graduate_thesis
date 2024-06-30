import os
from glob import glob
from multiprocessing import Pool

import numpy as np
from cv2 import (
  COLOR_GRAY2BGR,
  IMREAD_GRAYSCALE,
  NORM_L2,
  TERM_CRITERIA_COUNT,
  TERM_CRITERIA_EPS,
  calibrateCamera,
  cornerSubPix,
  cvtColor,
  drawChessboardCorners,
  findChessboardCorners,
  imread,
  imwrite,
  norm,
  projectPoints,
  resize,
)

threads_num = 12
img_mask = 'frames/*.jpg'
vis_dir = './debug'
pattern_size = (5, 8)
img_names = glob(img_mask)

found_chessboards = glob('debug/*.jpg')
found_chessboards = [i.replace('debug', 'frames') for i in found_chessboards]
img_names = [i for i in img_names if i in found_chessboards]

obj_point = np.zeros((np.prod(pattern_size), 3), np.float32)
obj_point[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

h, w = imread(img_names[0], IMREAD_GRAYSCALE).shape[:2]
print('h =', h, '\nw =', w)


def splitfn(fn):
  path, fn = os.path.split(fn)
  name, ext = os.path.splitext(fn)
  return path, name, ext


def get_corners(fn):
  img = imread(fn, IMREAD_GRAYSCALE)
  img = resize(img, (w, h))
  if img is None:
    print('Failed to load', fn)
    return None
  found, corners = findChessboardCorners(img, pattern_size)
  if found:
    term = (TERM_CRITERIA_EPS + TERM_CRITERIA_COUNT, 30, 0.1)
    cornerSubPix(img, corners, (5, 5), (-1, -1), term)
    if vis_dir:
      vis = cvtColor(img, COLOR_GRAY2BGR)
      drawChessboardCorners(vis, pattern_size, corners, found)
      name = splitfn(fn)[1]
      outfile = os.path.join(vis_dir, f'{name}.jpg')
      imwrite(outfile, vis)
  if not found:
    print(fn)
    return
  print(fn, 'OK')
  return corners.reshape(-1, 2)


# chessboards = [calib(x) for x in img_names]
pool = Pool(threads_num)
img_points = pool.map(get_corners, img_names)
img_points = [x for x in img_points if x is not None]
print(len(img_points), 'chessboards found')


def calibrate(img_points: list[np.ndarray]):
  total = len(img_points)

  rms, cam_mtx, dist_coefs, rvecs, tvecs = calibrateCamera(
    [obj_point] * total,
    img_points,
    (w, h),
    None,
    None,
  )
  print('\nRMS:', rms)

  errors = np.array([])
  for i in range(total):
    imgpoints2, _ = projectPoints(obj_point, rvecs[i], tvecs[i], cam_mtx, dist_coefs)
    error = norm(img_points[i], np.squeeze(imgpoints2), NORM_L2) / len(imgpoints2)
    errors = np.append(errors, error)

  print(errors)
  print(f'Min error: {errors.min()}')
  print(f'Max error: {errors.max()}')
  print(f'Mean error: {errors.mean()}')

  return cam_mtx, dist_coefs, errors


_, _, errors = calibrate(img_points)

new_img_points = [img_points[i] for i in np.where(errors < 2)[0]]

cam_mtx, dist_coefs, _ = calibrate(new_img_points)

print('camera matrix:\n', cam_mtx)
print('distortion coefficients: ', dist_coefs)

np.save('cam_mtx.npy', cam_mtx)
np.save('dist_coefs.npy', dist_coefs)
