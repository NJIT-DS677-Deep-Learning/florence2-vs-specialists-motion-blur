"""Map Florence-2 object labels to COCO category names."""

COCO_ALIASES = {
    "computer keyboard": "keyboard",
    "computer mouse": "mouse",
    "television": "tv",
    "doughnut": "donut",
    "mobile phone": "cell phone",
    "coffee cup": "cup",
    "cattle": "cow",
    "taxi": "car",
    "ski": "skis",
}


def normalize_category(label):
    """Normalize a Florence-2 label for COCO evaluation."""
    name = label.strip().lower()
    return COCO_ALIASES.get(name, name)
