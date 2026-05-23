# Project Description

AimVision is a portfolio-ready computer vision benchmark for offline FPS-style gameplay analysis.  
It combines object detection and multi-object tracking into a reproducible evaluation workflow.

## Problem Statement
Given recorded gameplay videos, detect target entities and maintain their identities across frames under:
- fast motion,
- partial occlusions,
- appearance changes,
- variable video quality.

## Target Classes
- `enemy`
- `teammate`
- `weapon`
- `head`

## Supported Inputs
- Offline video files (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`)
- Screenshots/frame datasets
- Synthetic aim-trainer clips

## Outputs
- Annotated videos with boxes, labels, confidences, tracking IDs, trajectories
- Per-frame tracking logs in CSV
- Tracker comparison reports in Markdown/CSV

## Scope and Restrictions
- Offline analysis only
- No interaction with running game process
- No memory reading, input control, DLL injection, or anti-cheat bypass

## Pipeline
1. Frame extraction and dataset preparation
2. Manual or assisted annotation (YOLO format)
3. YOLO detector training
4. Tracking-by-detection with multiple trackers
5. Benchmarking on shared video set
6. Metrics aggregation and report generation
