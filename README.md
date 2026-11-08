# Florence-2 vs. Specialist Models Under Motion Blur

**DS 677 – Deep Learning · NJIT · Industry Application Track - Vivek Shah and Victor Figueredo**

## Claim Statement

> We claim that Florence-2-base-ft's relative performance drop, (clean score − degraded score) ÷ clean score, is smaller than that of single-task models as motion blur increases, and we will test it against a BLIP-base baseline for image captioning (CIDEr, MS COCO Karpathy test split) and a DETR-ResNet-50 baseline for object detection (mAP, MS COCO val2017).

## Motivation

This project comes out of an internship with a nonprofit that is building an app of tools and resources for people with disabilities. One candidate feature is an **AI visual assistant**. A user with a visual impairment points their phone camera at their surroundings, the app captures a frame every few seconds, and a vision model describes the scene and the objects in it aloud.

Frames captured this way are rarely clean. A moving camera produces **motion blur**, and a user who cannot see the screen cannot retake a bad shot. A model is only useful to this audience if its output stays accurate under these conditions, because a confident but wrong description is worse than none.

Florence-2 is attractive for this app because a single 0.23B-parameter model handles captioning, object detection and OCR through text prompts. This project tests whether that unified model degrades more gracefully under motion blur than **specialist models built for one task each**.

## Repository Layout

```
README.md            # claim statement, links, how to run
report/              # report drafts and final PDF source
src/                 # data loading, degradation, inference, evaluation
experiments/         # configs, result CSVs, plots
└── wulver/          # Wulver Slurm job scripts and logs
demo/                # self-contained demo notebook
```
