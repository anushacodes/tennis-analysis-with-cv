import cv2
from ultralytics import YOLO

from player_reference import PlayerTracker


MODEL_PATH = "weights/best.pt"
PLAYER_CLASS = 15  # update to match your dataset


def run_video(video_path: str) -> None:
    model = YOLO(MODEL_PATH)

    # create player tracker
    tracker = PlayerTracker(smoothing=0.7, max_missing=15)

    cap = cv2.VideoCapture(video_path)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # run yolo on current frame
        results = model(frame, conf=0.25, verbose=False)[0]
        detections = results.boxes

        # collect all player boxes
        player_boxes = []
        for b in detections:
            if b.cls.numel() == 0:
                continue

            cls_id = int(b.cls[0])
            if cls_id != PLAYER_CLASS:
                continue

            x1, y1, x2, y2 = b.xyxy[0]
            player_boxes.append([
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            ])

        # update tracker with current detections
        tracker.update(player_boxes)
        boxes_by_id = tracker.get_boxes()

        # draw dynamic bounding boxes for each player
        for pid, box in boxes_by_id.items():
            if box is None:
                continue

            x1, y1, x2, y2 = box
            x1_i, y1_i = int(x1), int(y1)
            x2_i, y2_i = int(x2), int(y2)

            # choose color per player id
            if pid == 0:
                color = (0, 255, 0)   # player 0: green
                label = "player top"
            else:
                color = (255, 0, 0)   # player 1: blue
                label = "player bottom"

            # draw rectangle
            cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, 3)

            # draw label above box
            cv2.putText(
                frame,
                label,
                (x1_i, max(0, y1_i - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        # draw raw detection centers for debugging
        for b in detections:
            x1, y1, x2, y2 = b.xyxy[0]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

        cv2.imshow("dynamic player bounding boxes", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_video("dataset/videos/9.mp4")
