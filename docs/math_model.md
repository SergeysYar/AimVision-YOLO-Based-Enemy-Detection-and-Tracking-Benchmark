# Mathematical Model

## 1. Object Detection Formulation
Given image \(I \in \mathbb{R}^{H \times W \times 3}\), detector \(f_\theta\) predicts:
- class probability vector \(p(c \mid I)\),
- bounding box \(b = (x_1, y_1, x_2, y_2)\),
- confidence \(s \in [0,1]\).

A detection is:
\[
d_i = (b_i, c_i, s_i)
\]

## 2. Bounding Box Representation
Two common forms:
- Corner form: \((x_1, y_1, x_2, y_2)\)
- Center form: \((x_c, y_c, w, h)\)

Conversion:
\[
x_c = \frac{x_1 + x_2}{2},\quad y_c = \frac{y_1 + y_2}{2},\quad w = x_2 - x_1,\quad h = y_2 - y_1
\]

## 3. Confidence Score
Typical detector confidence:
\[
s = p(\text{object}) \cdot p(c \mid \text{object})
\]
Detections below threshold \(\tau\) are removed.

## 4. Intersection over Union (IoU)
For boxes \(A\) and \(B\):
\[
\text{IoU}(A,B)=\frac{|A \cap B|}{|A \cup B|}
\]
Higher IoU indicates stronger overlap.

## 5. Non-Maximum Suppression (NMS)
Sort detections by confidence, keep highest score box, suppress others with IoU above threshold \(\tau_{nms}\). Repeat until done.

## 6. Tracking-by-Detection
At frame \(t\):
1. Run detector to get \(D_t=\{d_i^t\}\).
2. Predict existing tracks \(\hat{T}_t\).
3. Match detections and tracks.
4. Update matched tracks, create new tracks for unmatched detections, age unmatched tracks.

## 7. Kalman Filter Idea (SORT)
Track state can include:
\[
\mathbf{x}_t = [x_c, y_c, w, h, \dot{x}_c, \dot{y}_c, \dot{w}, \dot{h}]^\top
\]
Prediction:
\[
\mathbf{x}_{t|t-1} = \mathbf{F}\mathbf{x}_{t-1|t-1}
\]
Update with measurement \(\mathbf{z}_t\) from detection using Kalman gain \(K_t\).

## 8. Hungarian Assignment
Build cost matrix \(C\) between predicted tracks and detections. Example:
\[
C_{ij} = 1 - \text{IoU}(\hat{b}_i, b_j)
\]
Solve minimum-cost bipartite matching with Hungarian algorithm.

## 9. Velocity Prediction
Simple constant velocity:
\[
\hat{x}_{c,t} = x_{c,t-1} + \dot{x}_{c,t-1},\quad \hat{y}_{c,t} = y_{c,t-1} + \dot{y}_{c,t-1}
\]
This stabilizes identity over short occlusions.

## 10. Assignment Cost Variants
Common costs:
\[
C_{ij}^{\text{IoU}} = 1-\text{IoU}_{ij},\quad
C_{ij}^{\text{dist}} = \lVert \mathbf{m}_i - \mathbf{m}_j \rVert_2
\]
where \(\mathbf{m}\) is center point. Custom trackers often use distance-only association.
