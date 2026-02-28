import argparse
from typing import List, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="4-point homography calibration (image -> robot ground plane)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--outfile", type=str, default="img_to_robot_h.npy")
    p.add_argument("--sandbox-w", type=float, required=True, help="Sandbox width in meters (x direction)")
    p.add_argument("--sandbox-h", type=float, required=True, help="Sandbox height in meters (y direction)")
    p.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "dshow", "msmf"],
        help="VideoCapture backend (Windows: dshow/msmf).",
    )
    return p.parse_args()


def draw_points(img: np.ndarray, pts: List[Tuple[int, int]]) -> np.ndarray:
    out = img.copy()
    for i, (x, y) in enumerate(pts):
        cv2.circle(out, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(out, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def backend_flag(name: str) -> int:
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def main() -> None:
    args = parse_args()

    print(
        f"Starting calibration with camera={args.camera}, backend={args.backend}, "
        f"size={args.width}x{args.height}",
        flush=True,
    )
    cap = cv2.VideoCapture(args.camera, backend_flag(args.backend))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    clicked: List[Tuple[int, int]] = []
    frame_ref = {"img": None}

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append((x, y))

    cv2.namedWindow("calib")
    cv2.setMouseCallback("calib", on_mouse)

    print("Click 4 sandbox corners in this order:")
    print("1) (0,0)  2) (W,0)  3) (W,H)  4) (0,H)")
    print("Keys: r=reset, s=save (when 4 points), ESC=quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera read failed.")

            frame_ref["img"] = frame
            vis = draw_points(frame, clicked)
            cv2.imshow("calib", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("r"):
                clicked.clear()
                print("Points reset.")
            if key == ord("s"):
                if len(clicked) != 4:
                    print("Need exactly 4 points.")
                    continue
                img_pts = np.array(clicked, dtype=np.float32)
                ground_pts = np.array(
                    [
                        [0.0, 0.0],
                        [args.sandbox_w, 0.0],
                        [args.sandbox_w, args.sandbox_h],
                        [0.0, args.sandbox_h],
                    ],
                    dtype=np.float32,
                )
                h, _ = cv2.findHomography(img_pts, ground_pts, method=0)
                if h is None:
                    print("Homography solve failed.")
                    continue
                np.save(args.outfile, h.astype(np.float32))
                print(f"Saved homography to {args.outfile}")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
