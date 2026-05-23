# Project Description

AimVision is an offline computer vision benchmark project for FPS-style gameplay footage. It detects entities (enemy, teammate, weapon, head), tracks them over time, compares tracker behavior across videos, and produces reproducible reports.

Scope:
- Offline videos, screenshots, and synthetic sandbox clips only.
- No interaction with running game processes.
- No automation, cheating, or input control.

Pipeline:
1. Dataset preparation from recorded clips.
2. YOLO detector training.
3. Detection + multi-tracker benchmarking.
4. Metrics aggregation and report generation.
