# Score model outputs the same way for every model.
# Captioning: CIDEr and BLEU-4 against the human reference captions (pycocoevalcap; needs Java)
# Detection:  COCO mAP against the ground-truth boxes (pycocotools)

import csv
import os

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.cocoeval import COCOeval

from src.config import RESULTS_COLUMNS, RESULTS_DIR

# Stop Java from printing warnings into the tokenizer's output. 
# Extra lines shift every caption onto the wrong image.
os.environ["JAVA_TOOL_OPTIONS"] = (os.environ.get("JAVA_TOOL_OPTIONS", "") + " -Xlog:disable").strip()


# Captioning
def _tokenize(captions):
  # Tokenize {image_id: [caption, ...]} with the official PTB tokenizer, and check nothing got shifted.
  data = {"__first__": [{"caption": "alignment check one"}]}
  for i, caps in captions.items():
    # Collapse line breaks and extra spaces, "." stands in for an empty caption
    data[i] = [{"caption": " ".join(c.split()) or "."} for c in caps]
  data["__last__"] = [{"caption": "alignment check two"}]

  out = PTBTokenizer().tokenize(data)
  if out.get("__first__") != ["alignment check one"] or out.get("__last__") != ["alignment check two"]:
    raise RuntimeError("Tokenizer output is misaligned, so captions would be scored against the wrong images.")

  # Remove the two check captions before scoring
  del out["__first__"], out["__last__"]
  return out


def score_captions(predictions, references):
  # Score captions
  # predictions: {image_id: "one predicted caption"}
  # references:  {image_id: ["human caption 1", "human caption 2", ...]}
  # Returns {"CIDEr": ..., "BLEU-4": ...} on the usual 0-100+ scale (e.g. CIDEr 140.0).
  
  # Score only images that have both a prediction and references
  ids = [i for i in predictions if i in references]

  # The official tokenizer lowercases captions and strips punctuation before scoring
  res = _tokenize({i: [predictions[i]] for i in ids})
  gts = _tokenize({i: references[i] for i in ids})

  # CIDEr (the COCO version, CIDEr-D) and BLEU-1 to BLEU-4
  cider, _ = Cider().compute_score(gts, res)
  bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)

  # Multiply by 100 to match the numbers reported in papers
  return {"CIDEr": float(cider) * 100, "BLEU-4": float(bleu[3]) * 100}


# Detection
def to_coco_box(box):
  # Convert a box from [x1, y1, x2, y2] (model output) to [x, y, width, height] (COCO format).
  x1, y1, x2, y2 = [float(v) for v in box]
  return [x1, y1, x2 - x1, y2 - y1]


def score_detections(detections, coco, image_ids):
  # Score detections with the official COCO evaluation.

  # detections: list of {"image_id", "category_id", "bbox": [x, y, w, h], "score"}
  # coco: the ground-truth object returned by load_val2017()
  # image_ids: the images that were run (so a small test only counts those images)
  # Returns mAP, mAP@0.5 and AP for small, medium and large objects, on a 0-100 scale.
  
  # No detections at all scores zero (pycocotools cannot load an empty list)
  if not detections:
    return {"mAP": 0.0, "mAP50": 0.0, "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0}

  # Run the standard COCO box evaluation on just these images
  evaluator = COCOeval(coco, coco.loadRes(detections), "bbox")
  evaluator.params.imgIds = list(image_ids)
  evaluator.evaluate()
  evaluator.accumulate()
  evaluator.summarize()

  # Stats holds the 12 standard COCO numbers; -1 means "no objects of that size", reported as NaN
  s = [float(v) * 100 if v >= 0 else float("nan") for v in evaluator.stats]
  return {"mAP": s[0], "mAP50": s[1], "AP_small": s[3], "AP_medium": s[4], "AP_large": s[5]}


# Results File
def save_result(row, path=os.path.join(RESULTS_DIR, "results.csv")):
  # Add one row to the results CSV, using the agreed columns from config.py.
  # Create the folder if needed, and write the header the first time
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  new_file = not os.path.exists(path)
  with open(path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, lineterminator="\n")
    if new_file:
      writer.writeheader()
    # Missing columns are left blank instead of causing an error
    writer.writerow({col: row.get(col, "") for col in RESULTS_COLUMNS})