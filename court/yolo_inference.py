import cv2
import numpy as np
from ultralytics import YOLO

from court_reference import CourtReference

# load model and court reference
model = YOLO("weights/best.pt")
court_ref = CourtReference()

# mapping yolo IDs → reference IDs
REMAPPING = {
    0: 0,   
    6: 1,   
    7: 2,   
    8: 3,   
    9: 4,   
    10: 5,  
    11: 6,  
    12: 7,  
    13: 8,  
    1: 9,   
    2: 10,  
    3: 11,  
    4: 12,  
    5: 13   
}

BALL_CLASS_ID = 14      # your dataset says class 14 = ball
PLAYER_CLASS_ID = 15    # your dataset says class 15 = player
def get_homography(boxes, frame_shape):
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

    # SAFETY 1: Require at least 5 points, not 4.
    # 4 points is the mathematical minimum, but it is extremely unstable.
    if len(ref_pts) < 5:
        return None

    ref_pts = np.array(ref_pts, dtype=np.float32).reshape(-1, 1, 2)
    img_pts = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)

    # SAFETY 2: Use RANSAC with a stricter threshold (e.g., 5.0)
    H, mask = cv2.findHomography(ref_pts, img_pts, cv2.RANSAC, 5.0)
    return H


def draw_projected_court(frame, H):
    if H is None:
        return

    lines = court_ref.get_court_lines()

    pts = []
    for p1, p2 in lines:
        pts.append(p1)
        pts.append(p2)

    pts = np.array([pts], dtype=np.float32)
    
    # transform normalized coords -> pixel coords
    try:
        warped = cv2.perspectiveTransform(pts, H)[0]
    except Exception as e:
        # If matrix is invalid, perspectiveTransform might fail
        return

    for i in range(0, len(warped), 2):
        x1, y1 = warped[i]
        x2, y2 = warped[i + 1]

        # SAFETY 3: Check for NaN/Inf and Integer Overflow
        # If the projection is garbage, x1/y1 can be NaN or 100000000
        if (np.isnan(x1) or np.isnan(y1) or 
            np.isnan(x2) or np.isnan(y2) or 
            abs(x1) > 10000 or abs(y1) > 10000):
            continue

        # Safe to cast to int now
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)



def run_video(video_path):
    cap = cv2.VideoCapture(video_path)
    last_H = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.25, verbose=False)
        boxes = results[0].boxes

        # update homography
        H = get_homography(boxes, frame.shape)
        if H is not None:
            last_H = H

        # draw court
        if last_H is not None:
            draw_projected_court(frame, last_H)

        for b in boxes:
            cls = int(b.cls[0])
            x1, y1, x2, y2 = b.xyxy[0]

            # BALL → cyan box
            if cls == BALL_CLASS_ID:
                cv2.rectangle(frame,
                              (int(x1), int(y1)), (int(x2), int(y2)),
                              (255, 255, 0), 3)   # CYAN
                cv2.putText(frame, "BALL",
                            (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 0), 2)
                continue

            # PLAYER → blue dot
            if cls == PLAYER_CLASS_ID:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.circle(frame, (cx, cy), 7, (255, 0, 0), -1)
                continue

            # COURT KEYPOINT → green dot
            if cls in REMAPPING:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        cv2.imshow("Tennis Court Homography + Ball Tracking", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_video("dataset/videos/5.mp4")
