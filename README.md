# 🎾 Tennis Analysis with Computer Vision

This project implements a single-camera tennis analysis pipeline that detects and tracks **court lines, players, and the tennis ball** from match videos. The goal is to provide a lightweight, accessible alternative to expensive multi-camera systems while maintaining strong accuracy and visual consistency.

The system combines **YOLO-based detection**, **TrackNet ball tracking**, and **geometric court reconstruction** to produce annotated videos with real-time overlays.

---


## Key Features

* **Court line detection & reconstruction**

  * YOLO keypoint detection for 14 tennis court landmarks
  * Homography estimation for consistent court mapping
    
* **Player detection & tracking**

  * YOLO-based player detection
  * Stable player ID assignment across frames
    
* **Ball tracking**

  * TrackNet heatmap-based ball localization
  * Temporal smoothing and outlier filtering
    
* **End-to-end video inference**

  * Single input video → fully annotated output video
  * Optional trajectory extrapolation for missed detections

---


## Models Used

* **YOLO (Ultralytics)**

  * Court keypoints
  * Players
  * ball detection for validation
    
* **TrackNet**

  * High-speed tennis ball tracking via heatmap regression

---


## Project Structure

```text
.
├── court/
│   ├── court_reference.py          # court geometry & homography
│   └── final_inference.py          # end-to-end inference pipeline
├── player_reference.py             # player tracking logic
├── tracknet_model.py               # tracknet architecture
├── train_yolo.py                   # yolo training script
├── yolo_and_tracknet_inference.py  # combined inference & visualization
├── weights/                        # trained model weights
├── dataset/                        # videos & annotations
└── output/                         # generated output videos
```

---

## Environment Setup

```bash
conda create -n tennis-env python=3.10 -y
conda activate tennis-env
pip install -r requirements.txt
```

---

## Running Inference

```bash
python court/final_inference.py \
  --video_path dataset/videos/8.mp4 \
  --tracknet_model weights/tracknet.pt \
  --yolo_model weights/best.pt \
  --output_path output/output8.mp4 \
  --ball_smoothing 0.5 \
  --trace_length 7
```

---

## Performance Highlights

* YOLO court keypoint detection achieves **~0.98 mAP@50**
* Stable court reconstruction even under partial occlusions
* Robust ball tracking during high-speed rallies

---

## Notes

This project was developed as part of **CS673 – Computer Vision** and is designed to be **modular, extensible, and reproducible**. Each component (court detection, player tracking, ball tracking) can be independently improved or replaced without affecting the overall pipeline.


