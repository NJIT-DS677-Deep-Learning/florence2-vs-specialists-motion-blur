"""Load the two evaluation datasets.

Captioning: MS COCO Karpathy test split (5,000 images, 5+ human captions each)
Detection:  MS COCO val2017 (5,000 images with ground-truth boxes)
"""
import json
import os

from PIL import Image
from pycocotools.coco import COCO

from src.config import DATA_DIR


def open_image(path):
  """Open an image as RGB. Some COCO images are grayscale, and the models expect RGB."""
  return Image.open(path).convert("RGB")


def load_karpathy_test(data_dir=DATA_DIR):
  """Return a list of dicts, one per test image: image_id, path, references."""
  with open(os.path.join(data_dir, "dataset_coco.json")) as f:
    images = json.load(f)["images"]

  items = []
  for im in images:
    if im["split"] == "test":
      items.append({
        "image_id": im["cocoid"],
        "path": os.path.join(data_dir, im["filepath"], im["filename"]),
        "references": [s["raw"] for s in im["sentences"]]
      })
  return items


def load_val2017(data_dir=DATA_DIR):
  """Return (items, coco).

  items: a list of dicts, one per image: image_id, path
  coco:  the pycocotools object holding the ground-truth boxes (used later for mAP)
  """
  coco = COCO(os.path.join(data_dir, "annotations", "instances_val2017.json"))

  items = []
  for image_id in sorted(coco.getImgIds()):
    file_name = coco.imgs[image_id]["file_name"]
    items.append({
      "image_id": image_id,
      "path": os.path.join(data_dir, "val2017", file_name)
    })
  return items, coco
