# Mathematical Model

This version avoids LaTeX-only syntax and uses GitHub-friendly plain formulas.

## 1. Object Detection Formulation
Input image:
- `I` with shape `H x W x 3`

Detector output for object `i`:
- `b_i = (x1, y1, x2, y2)` (bounding box)
- `c_i` (class id)
- `s_i` (confidence, from 0 to 1)

Detection tuple:
```text
d_i = (b_i, c_i, s_i)
```

## 2. Bounding Box Representation
Two common formats:
- Corner format: `(x1, y1, x2, y2)`
- Center format: `(xc, yc, w, h)`

Conversion:
```text
xc = (x1 + x2) / 2
yc = (y1 + y2) / 2
w  = x2 - x1
h  = y2 - y1
```

## 3. Confidence Score
Typical interpretation:
```text
s = P(object) * P(class | object)
```
Detections with `s < threshold` are filtered out.

## 4. IoU (Intersection over Union)
For boxes `A` and `B`:
```text
IoU(A, B) = area(A ∩ B) / area(A ∪ B)
```
`IoU` is in `[0, 1]`. Higher means better overlap.

## 5. Non-Maximum Suppression (NMS)
1. Sort detections by confidence.
2. Keep highest-confidence box.
3. Remove boxes with IoU above `nms_threshold`.
4. Repeat.

## 6. Tracking-by-Detection Pipeline
For frame `t`:
1. Run detector and get `D_t`.
2. Predict current track states.
3. Match detections to tracks.
4. Update matched tracks.
5. Create new tracks for unmatched detections.
6. Age and remove stale tracks.

## 7. Kalman Filter Idea (SORT)
State vector example:
```text
x_t = [xc, yc, w, h, vxc, vyc, vw, vh]
```
Prediction:
```text
x_t_pred = F * x_(t-1)
```
Then correct prediction with measurement from detector.

## 8. Hungarian Matching
Cost matrix between predicted tracks `i` and detections `j`:
```text
C[i, j] = 1 - IoU(track_i, det_j)
```
Hungarian algorithm finds minimal total matching cost.

## 9. Constant Velocity Prediction
Simple motion model:
```text
xc_t = xc_(t-1) + vxc_(t-1)
yc_t = yc_(t-1) + vyc_(t-1)
```
This helps preserve identity through short occlusions.

## 10. Assignment Cost Variants
IoU-based:
```text
C_iou = 1 - IoU
```
Distance-based:
```text
C_dist = sqrt((x1 - x2)^2 + (y1 - y2)^2)
```
Custom lightweight trackers often rely on center-distance cost.
