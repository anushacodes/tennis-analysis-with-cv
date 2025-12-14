import torch
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from itertools import groupby
from scipy.spatial import distance

from ultralytics import YOLO

from tracknet_model import BallTrackerNet
from court_reference import CourtReference
from player_reference import PlayerTracker



def postprocess(mask):
    """
    mask: numpy array with shape (1, H*W)
    Returns (x, y) or (None, None)
    """
    mask = mask.reshape(360, 640)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None
    return int(np.mean(xs)), int(np.mean(ys))


def infer_tracknet(frames, model, device):
    """Run TrackNet on all frames to get ball positions"""
    height = 360
    width = 640

    dists = [-1, -1]
    ball_track = [(None, None), (None, None)]

    print("Running TrackNet ball detection...")
    for num in tqdm(range(2, len(frames))):
        img = cv2.resize(frames[num], (width, height))
        img_prev = cv2.resize(frames[num - 1], (width, height))
        img_preprev = cv2.resize(frames[num - 2], (width, height))

        imgs = np.concatenate([img, img_prev, img_preprev], axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.transpose(imgs, (2, 0, 1))

        inp = torch.from_numpy(imgs).float().unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp, testing=True)
            out = out.argmax(dim=1).cpu().numpy()

        x_pred, y_pred = postprocess(out)
        ball_track.append((x_pred, y_pred))

        if ball_track[-1][0] and ball_track[-2][0]:
            dist = distance.euclidean(ball_track[-1], ball_track[-2])
        else:
            dist = -1
        dists.append(dist)

    return ball_track, dists


def remove_outliers(ball_track, dists, max_dist=100):
    """Remove outlier ball detections"""
    outliers = np.where(np.array(dists) > max_dist)[0]

    for i in outliers:
        if i + 1 < len(dists) and (dists[i + 1] > max_dist or dists[i + 1] == -1):
            ball_track[i] = (None, None)
        elif i > 0 and dists[i - 1] == -1:
            ball_track[i - 1] = (None, None)

    return ball_track


def split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):
    """Split ball track into segments"""
    list_det = [0 if p[0] else 1 for p in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor = 0
    min_value = 0
    result = []

    for i, (k, l) in enumerate(groups):
        if (k == 1) and (0 < i < len(groups) - 1):
            dist = distance.euclidean(ball_track[cursor - 1], ball_track[cursor + l])
            if (l >= max_gap) or (dist / l > max_dist_gap):
                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                min_value = cursor + l - 1
        cursor += l

    if len(list_det) - min_value > min_track:
        result.append([min_value, len(list_det)])

    return result


def interpolation(coords):
    """Interpolate missing ball positions"""
    def nan_helper(arr):
        return np.isnan(arr), lambda z: z.nonzero()[0]

    x = np.array([c[0] if c[0] is not None else np.nan for c in coords])
    y = np.array([c[1] if c[1] is not None else np.nan for c in coords])

    nans_x, idx = nan_helper(x)
    x[nans_x] = np.interp(idx(nans_x), idx(~nans_x), x[~nans_x])

    nans_y, idy = nan_helper(y)
    y[nans_y] = np.interp(idy(nans_y), idy(~nans_y), y[~nans_y])

    return list(zip(x, y))


REMAPPING = {
    0: 0, 6: 1, 7: 2, 8: 3, 9: 4, 10: 5, 11: 6, 12: 7, 13: 8,
    1: 9, 2: 10, 3: 11, 4: 12, 5: 13
}

BALL_CLASS_ID = 14
PLAYER_CLASS_ID = 15


def get_homography(boxes, frame_shape, court_ref):
    """Compute homography from court keypoints"""
    img_h, img_w = frame_shape[:2]

    ref_pts = []
    img_pts = []

    for b in boxes:
        if b.cls.numel() == 0:
            continue

        cls_id = int(b.cls[0])
        if cls_id not in REMAPPING:
            continue

        mapped_id = REMAPPING[cls_id]

        x1, y1, x2, y2 = b.xyxy[0]
        cx = float((x1 + x2) / 2)
        cy = float((y1 + y2) / 2)

        img_pts.append([cx, cy])
        ref_pts.append(list(court_ref.court_conf[mapped_id]))

    if len(ref_pts) < 5:
        return None

    ref_pts = np.array(ref_pts, dtype=np.float32).reshape(-1, 1, 2)
    img_pts = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(ref_pts, img_pts, cv2.RANSAC, 5.0)
    return H


def draw_projected_court(frame, H, court_ref, alpha=0.6):
    """Draw court lines with transparency"""
    if H is None:
        return

    lines = court_ref.get_court_lines()
    overlay = frame.copy()

    pts = []
    for p1, p2 in lines:
        pts.append(p1)
        pts.append(p2)

    pts = np.array([pts], dtype=np.float32)

    try:
        warped = cv2.perspectiveTransform(pts, H)[0]
    except:
        return

    for i in range(0, len(warped), 2):
        x1, y1 = warped[i]
        x2, y2 = warped[i + 1]

        if (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2) or
            abs(x1) > 10000 or abs(y1) > 10000):
            continue

        cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # red


    # Blend with alpha
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


class ExponentialSmoother:
    """Exponential moving average smoother"""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if new_value is None:
            return self.value

        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value

        return self.value


def read_video(path_video):
    """Load all frames from video"""
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    print("Loading video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    print(f"Loaded {len(frames)} frames at {fps} FPS")
    return frames, fps


def run_inference(video_path, tracknet_model_path, yolo_model_path, output_path, 
                  enable_interpolation=True, ball_smoothing=0.5, trace_length=7):
    """
    Main inference pipeline:
    1. Load video frames
    2. Run TrackNet for ball detection
    3. Run YOLO for court + players
    4. Overlay everything and save
    """
    
    device = torch.device("cpu")
    
    # Load TrackNet
    print("Loading TrackNet model...")
    tracknet_model = BallTrackerNet().to(device)
    state = torch.load(tracknet_model_path, map_location="cpu")
    tracknet_model.load_state_dict(state, strict=True)
    tracknet_model.eval()

    # Load YOLO
    print("Loading YOLO model...")
    yolo_model = YOLO(yolo_model_path)

    # Court reference
    court_ref = CourtReference()

    # Player tracker
    player_tracker = PlayerTracker(smoothing=0.7, max_missing=15)

    frames, fps = read_video(video_path)
    
    ball_track, dists = infer_tracknet(frames, tracknet_model, device)
    ball_track = remove_outliers(ball_track, dists)

    # Optional interpolation
    if enable_interpolation:
        print("Interpolating ball trajectory...")
        segments = split_track(ball_track)
        for start, end in segments:
            sub = interpolation(ball_track[start:end])
            ball_track[start:end] = sub

    # Ball position smoother (for display coordinates)
    ball_smoother_x = ExponentialSmoother(alpha=ball_smoothing)
    ball_smoother_y = ExponentialSmoother(alpha=ball_smoothing)

    # Process Video with YOLO + Overlay 
    print("Processing video with YOLO and overlaying...")
    
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    last_H = None

    for num in tqdm(range(len(frames))):
        frame = frames[num].copy()

        results = yolo_model(frame, conf=0.25, verbose=False)[0]
        boxes = results.boxes

        H = get_homography(boxes, frame.shape, court_ref)
        if H is not None:
            last_H = H

        if last_H is not None:
            draw_projected_court(frame, last_H, court_ref, alpha=0.5)

        player_boxes = []
        for b in boxes:
            if b.cls.numel() == 0:
                continue
            cls_id = int(b.cls[0])
            if cls_id != PLAYER_CLASS_ID:
                continue
            x1, y1, x2, y2 = b.xyxy[0]
            player_boxes.append([float(x1), float(y1), float(x2), float(y2)])

        player_tracker.update(player_boxes)
        boxes_by_id = player_tracker.get_boxes()

        for pid, box in boxes_by_id.items():
            if box is None:
                continue

            x1, y1, x2, y2 = box
            x1_i, y1_i = int(x1), int(y1)
            x2_i, y2_i = int(x2), int(y2)

            if pid == 0:
                color = (0, 200, 0)  # Green
                label = "Player Top"
            else:
                color = (200, 0, 200)  # Magenta
                label = "Player Bottom"

            # Draw rectangle with rounded corners effect (simple thick line)
            cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, 3)

            # Draw label with background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
            label_w, label_h = label_size
            
            # Background rectangle for text
            cv2.rectangle(
                frame,
                (x1_i, max(0, y1_i - label_h - 10)),
                (x1_i + label_w + 10, y1_i),
                color,
                -1
            )

            # Text
            cv2.putText(
                frame,
                label,
                (x1_i + 5, y1_i - 5),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # Ball Trail
        if num < len(ball_track):
            for i in range(trace_length):
                idx = num - i
                if idx < 0 or idx >= len(ball_track):
                    continue

                x_raw, y_raw = ball_track[idx]

                if x_raw is None or y_raw is None:
                    continue

                # Convert from 640x360 to original resolution
                x = int(x_raw * width / 640)
                y = int(y_raw * height / 360)

                # Apply smoothing only to current ball position
                if i == 0:
                    x = ball_smoother_x.update(x)
                    y = ball_smoother_y.update(y)
                    if x is not None and y is not None:
                        x = int(x)
                        y = int(y)

                # Draw trail with decreasing size
                radius = max(3, 8 - i)
                thickness = max(2, 12 - i * 2)
                
                # Gradient from bright yellow to red
                intensity = int(255 * (trace_length - i) / trace_length)
                color = (0, intensity, 255)  # Yellow to orange

                cv2.circle(frame, (x, y), radius, color, thickness)

        out.write(frame)

    out.release()
    print(f"Output saved to: {output_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Tennis Analysis Inference")
    parser.add_argument("--video_path", type=str, required=True, help="Input video path")
    parser.add_argument("--tracknet_model", type=str, required=True, help="TrackNet model path")
    parser.add_argument("--yolo_model", type=str, required=True, help="YOLO model path")
    parser.add_argument("--output_path", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--no_interpolation", action="store_true", help="Disable ball interpolation")
    parser.add_argument("--ball_smoothing", type=float, default=0.5, help="Ball smoothing factor (0-1)")
    parser.add_argument("--trace_length", type=int, default=7, help="Ball trail length")

    args = parser.parse_args()

    run_inference(
        video_path=args.video_path,
        tracknet_model_path=args.tracknet_model,
        yolo_model_path=args.yolo_model,
        output_path=args.output_path,
        enable_interpolation=not args.no_interpolation,
        ball_smoothing=args.ball_smoothing,
        trace_length=args.trace_length
    )

# example usage:
# python court/final_inference.py --video_path dataset/videos/5.mp4 --tracknet_model weights/tracknet.pt 
# --yolo_model weights/best.pt --output_path output5-new.mp4 --ball_smoothing 0.5 --trace_length 7