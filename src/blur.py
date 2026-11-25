# Motion blur for the robustness probe.

# Two blur types on one severity scale. blur_px is how far the camera moves while the photo is taken, in pixels:
#   linear: the camera moves in a straight line, in a random direction
#   shake:  the camera follows a random, curved path, like a shaking hand
# blur_px = 0 returns the image unchanged.
# The blur for each image is fixed by (seed, image_id), so every model sees exactly the same blurred image.

import numpy as np
from PIL import Image

BLUR_TYPES = ("linear", "shake")

def _linear_path(blur_px, rng):
  # A straight line of length blur_px through the center, at a random angle.
  angle = rng.uniform(0, np.pi)
  half = np.array([np.cos(angle), np.sin(angle)]) * blur_px / 2
  return np.stack([-half, half])


def _shake_path(blur_px, rng, steps=64):
  # A smooth random walk, scaled so its two farthest points are blur_px apart.
  velocity = rng.normal(size=2)
  points = [np.zeros(2)]
  for _ in range(steps - 1):
    # Momentum keeps the path smooth, like a real hand movement
    velocity = 0.7 * velocity + rng.normal(size=2)
    points.append(points[-1] + velocity)
  points = np.array(points)

  # Largest distance between any two points on the path
  spread = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)).max()
  return points * (blur_px / spread)


def _draw_kernel(path):
  # Draw the camera path into a blur kernel that sums to 1 and is centered (so the image is not shifted).
  # Sample the path densely, 4 samples per pixel of length
  samples = []
  for start, end in zip(path[:-1], path[1:]):
    n = max(2, int(np.ceil(np.linalg.norm(end - start) * 4)))
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    samples.append(start + t * (end - start))
  samples.append(path[-1:])
  samples = np.concatenate(samples)

  # Center the path on its average point, so blurring does not move objects (boxes stay aligned)
  samples -= samples.mean(axis=0)
  radius = int(np.ceil(np.abs(samples).max())) + 1
  kernel = np.zeros((2 * radius + 1, 2 * radius + 1))

  # Spread each sample over its 4 nearest pixels (keeps sub-pixel accuracy)
  x, y = samples[:, 0] + radius, samples[:, 1] + radius
  x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
  fx, fy = x - x0, y - y0
  np.add.at(kernel, (y0, x0), (1 - fx) * (1 - fy))
  np.add.at(kernel, (y0, x0 + 1), fx * (1 - fy))
  np.add.at(kernel, (y0 + 1, x0), (1 - fx) * fy)
  np.add.at(kernel, (y0 + 1, x0 + 1), fx * fy)
  return kernel / kernel.sum()


def make_kernel(blur_type, blur_px, seed, image_id):
  # Return the blur kernel for one image. The same (seed, image_id) always gives the same kernel.
  if blur_type not in BLUR_TYPES:
    raise ValueError(f"blur_type must be one of {BLUR_TYPES}, got {blur_type!r}")
  rng = np.random.default_rng([seed, image_id])
  path = _linear_path(blur_px, rng) if blur_type == "linear" else _shake_path(blur_px, rng)
  return _draw_kernel(path)


def _convolve(image_array, kernel):
  # Blur an H x W x 3 array with the kernel (FFT, with mirrored edges so borders do not darken).
  r = kernel.shape[0] // 2
  h, w = image_array.shape[:2]
  padded = np.pad(image_array, ((r, r), (r, r), (0, 0)), mode="reflect")

  # Place the kernel center at (0, 0) so the result is not shifted
  full = np.zeros(padded.shape[:2])
  full[:kernel.shape[0], :kernel.shape[1]] = kernel
  full = np.roll(full, (-r, -r), axis=(0, 1))

  blurred = np.fft.irfft2(np.fft.rfft2(padded, axes=(0, 1)) * np.fft.rfft2(full)[:, :, None], s=padded.shape[:2], axes=(0, 1))
  return blurred[r:r + h, r:r + w]


def apply_blur(image, blur_type, blur_px, seed, image_id):
  # Return a motion-blurred copy of a PIL image. blur_px = 0 returns the image unchanged.
  if blur_px == 0:
    return image
  kernel = make_kernel(blur_type, blur_px, seed, image_id)
  array = np.asarray(image.convert("RGB"), dtype=np.float64)
  blurred = _convolve(array, kernel)
  return Image.fromarray(np.clip(np.rint(blurred), 0, 255).astype(np.uint8))
