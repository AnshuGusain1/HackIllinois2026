import argparse
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraConfig:
    device_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    roi: Tuple[int, int, int, int] = (0, 0, 640, 480)  # x, y, w, h
    backend: str = "auto"


@dataclass
class PreprocessConfig:
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    shadow_blur_ksize: int = 41  # must be odd
    gamma: float = 0.9
    use_scharr: bool = False
    median_ksize: int = 3


@dataclass
class UnevennessConfig:
    grad_pctl: float = 85.0
    var_window: int = 9
    roughness_alpha: float = 0.2
    slope_min_grad: float = 20.0


@dataclass
class GridConfig:
    sandbox_w_m: float = 1.2
    sandbox_h_m: float = 0.8
    grid_w: int = 20
    grid_h: int = 20
    map_alpha: float = 0.2
    revisit_hi: float = 0.6
    revisit_lo: float = 0.4
    min_hits_for_revisit: int = 3
    sample_stride_px: int = 8


@dataclass
class IMUConfig:
    use_imu: bool = True
    pitch_roll_deg_gate: float = 12.0
    vib_ref: float = 0.8
    imu_alpha: float = 0.2


@dataclass
class PipelineState:
    roughness_ema: float = 0.0
    severity_ema: float = 0.0
    revisit_mask: np.ndarray = field(default_factory=lambda: np.zeros((20, 20), dtype=np.uint8))
    no_valid_projection_frames: int = 0
    last_valid_projection_count: int = 0
    last_updated_cells: int = 0


class SandUnevennessPipeline:
    def __init__(
        self,
        cam_cfg: CameraConfig,
        prep_cfg: PreprocessConfig,
        un_cfg: UnevennessConfig,
        grid_cfg: GridConfig,
        imu_cfg: IMUConfig,
        img_to_robot_h: np.ndarray,
    ):
        self.cam_cfg = cam_cfg
        self.prep_cfg = prep_cfg
        self.un_cfg = un_cfg
        self.grid_cfg = grid_cfg
        self.imu_cfg = imu_cfg

        self.cap = self._init_camera(cam_cfg)
        self.img_to_robot_h = img_to_robot_h.astype(np.float32)

        self.grid_score = np.zeros((grid_cfg.grid_h, grid_cfg.grid_w), dtype=np.float32)
        self.grid_hits = np.zeros_like(self.grid_score, dtype=np.int32)
        self.state = PipelineState(revisit_mask=np.zeros_like(self.grid_score, dtype=np.uint8))

    @staticmethod
    def _init_camera(cfg: CameraConfig) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(cfg.device_id, backend_flag(cfg.backend))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera.")
        return cap

    def capture_frame(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera read failed.")
        x, y, w, h = self.cam_cfg.roi
        return frame[y : y + h, x : x + w]

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=self.prep_cfg.clahe_clip,
            tileGridSize=(self.prep_cfg.clahe_tile, self.prep_cfg.clahe_tile),
        )
        eq = clahe.apply(gray)

        k = self.prep_cfg.shadow_blur_ksize
        if k % 2 == 0:
            k += 1
        illum = cv2.GaussianBlur(eq, (k, k), 0)
        flat = cv2.divide(eq, illum + 1, scale=128)

        f = np.clip(flat.astype(np.float32) / 255.0, 0.0, 1.0)
        f = np.power(f, self.prep_cfg.gamma)
        out = np.uint8(np.clip(f * 255.0, 0, 255))
        return out

    def compute_gradient_map(self, img_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.prep_cfg.use_scharr:
            gx = cv2.Scharr(img_u8, cv2.CV_32F, 1, 0)
            gy = cv2.Scharr(img_u8, cv2.CV_32F, 0, 1)
        else:
            gx = cv2.Sobel(img_u8, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img_u8, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.medianBlur(mag.astype(np.float32), self.prep_cfg.median_ksize)
        return mag, gx, gy

    def compute_roughness(self, grad_mag: np.ndarray, img_u8: np.ndarray) -> float:
        grad_score = float(np.percentile(grad_mag, self.un_cfg.grad_pctl))
        w = self.un_cfg.var_window
        i = img_u8.astype(np.float32)
        mean = cv2.boxFilter(i, -1, (w, w))
        sq_mean = cv2.boxFilter(i * i, -1, (w, w))
        var = np.maximum(sq_mean - mean * mean, 0.0)
        var_score = float(np.percentile(var, 80))

        raw = 0.7 * grad_score + 0.3 * var_score
        a = self.un_cfg.roughness_alpha
        self.state.roughness_ema = (1.0 - a) * self.state.roughness_ema + a * raw
        return self.state.roughness_ema

    def estimate_slope_direction_deg(self, gx: np.ndarray, gy: np.ndarray, grad_mag: np.ndarray) -> Optional[float]:
        mask = grad_mag > self.un_cfg.slope_min_grad
        if int(np.count_nonzero(mask)) < 50:
            return None
        wx = float(np.sum(gx[mask]))
        wy = float(np.sum(gy[mask]))
        if abs(wx) + abs(wy) < 1e-6:
            return None
        return math.degrees(math.atan2(wy, wx))

    def fuse_imu(self, roughness: float, roll_deg: float, pitch_deg: float, vib_energy: float) -> float:
        if not self.imu_cfg.use_imu:
            return roughness

        tilt = max(abs(roll_deg), abs(pitch_deg))
        tilt_penalty = 0.75 if tilt > self.imu_cfg.pitch_roll_deg_gate else 1.0
        vib_boost = 1.0 + 0.25 * np.clip(vib_energy / max(self.imu_cfg.vib_ref, 1e-3), 0.0, 2.0)

        severity_raw = roughness * tilt_penalty * float(vib_boost)
        a = self.imu_cfg.imu_alpha
        self.state.severity_ema = (1.0 - a) * self.state.severity_ema + a * severity_raw
        return self.state.severity_ema

    def project_img_to_robot(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ones = np.ones_like(u, dtype=np.float32)
        p_img = np.stack([u, v, ones], axis=0).astype(np.float32)  # 3xN
        p_r = self.img_to_robot_h @ p_img
        xr = p_r[0] / (p_r[2] + 1e-6)
        yr = p_r[1] / (p_r[2] + 1e-6)
        return xr, yr

    @staticmethod
    def robot_to_world(xr: np.ndarray, yr: np.ndarray, pose_xy_yaw: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
        x, y, yaw = pose_xy_yaw
        c, s = math.cos(yaw), math.sin(yaw)
        xw = x + c * xr - s * yr
        yw = y + s * xr + c * yr
        return xw, yw

    def update_grid_map(self, grad_mag: np.ndarray, severity: float, pose_xy_yaw: Tuple[float, float, float]) -> None:
        h, w = grad_mag.shape
        stride = self.grid_cfg.sample_stride_px
        us = np.arange(0, w, stride, dtype=np.float32)
        vs = np.arange(0, h, stride, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)
        u = uu.reshape(-1)
        v = vv.reshape(-1)

        gvals = grad_mag[v.astype(np.int32), u.astype(np.int32)]
        g95 = np.percentile(grad_mag, 95) + 1e-6
        g_norm = gvals / g95

        sev_norm = severity / (self.state.severity_ema + 1e-6)
        pt_score = np.clip(0.5 * g_norm + 0.5 * sev_norm, 0.0, 1.5)

        xr, yr = self.project_img_to_robot(u, v)
        xw, yw = self.robot_to_world(xr, yr, pose_xy_yaw)

        gx = np.floor((xw / self.grid_cfg.sandbox_w_m) * self.grid_cfg.grid_w).astype(np.int32)
        gy = np.floor((yw / self.grid_cfg.sandbox_h_m) * self.grid_cfg.grid_h).astype(np.int32)

        valid = (
            (gx >= 0)
            & (gx < self.grid_cfg.grid_w)
            & (gy >= 0)
            & (gy < self.grid_cfg.grid_h)
        )
        valid_count = int(np.count_nonzero(valid))
        self.state.last_valid_projection_count = valid_count
        if valid_count == 0:
            self.state.no_valid_projection_frames += 1
            self.state.last_updated_cells = 0
            if self.state.no_valid_projection_frames % 30 == 0:
                print(
                    "Warning: 0 projected points landed in sandbox grid. "
                    "Check homography (--hfile) and sandbox dimensions.",
                    flush=True,
                )
            return
        self.state.no_valid_projection_frames = 0
        gx = gx[valid]
        gy = gy[valid]
        scores = pt_score[valid].astype(np.float32)

        a = self.grid_cfg.map_alpha
        touched = set()
        for i in range(len(scores)):
            yy, xx = gy[i], gx[i]
            old = self.grid_score[yy, xx]
            self.grid_score[yy, xx] = (1.0 - a) * old + a * scores[i]
            self.grid_hits[yy, xx] += 1
            touched.add((yy, xx))
        self.state.last_updated_cells = len(touched)

    def compute_revisit_targets(self) -> List[Tuple[int, int]]:
        high = (self.grid_score >= self.grid_cfg.revisit_hi) & (self.grid_hits >= self.grid_cfg.min_hits_for_revisit)
        keep = (self.state.revisit_mask == 1) & (self.grid_score >= self.grid_cfg.revisit_lo)
        self.state.revisit_mask = (high | keep).astype(np.uint8)

        ys, xs = np.where(self.state.revisit_mask == 1)
        return list(zip(xs.tolist(), ys.tolist()))

    def process_one_frame(
        self,
        pose_xy_yaw: Tuple[float, float, float],
        imu_roll_pitch_vib: Tuple[float, float, float],
    ) -> Dict[str, object]:
        frame = self.capture_frame()
        pre = self.preprocess(frame)
        grad_mag, gx, gy = self.compute_gradient_map(pre)
        roughness = self.compute_roughness(grad_mag, pre)
        slope_dir = self.estimate_slope_direction_deg(gx, gy, grad_mag)

        roll, pitch, vib = imu_roll_pitch_vib
        severity = self.fuse_imu(roughness, roll, pitch, vib)

        self.update_grid_map(grad_mag, severity, pose_xy_yaw)
        revisit = self.compute_revisit_targets()

        return {
            "frame": frame,
            "preprocessed": pre,
            "gradient_map": grad_mag,
            "roughness_score": float(roughness),
            "slope_direction_deg": slope_dir,
            "terrain_severity": float(severity),
            "grid_score": self.grid_score.copy(),
            "revisit_cells": revisit,
        }

    def close(self) -> None:
        self.cap.release()


def quick_homography_from_4_corners(img_pts: np.ndarray, ground_pts_m: np.ndarray) -> np.ndarray:
    if img_pts.shape != (4, 2) or ground_pts_m.shape != (4, 2):
        raise ValueError("img_pts and ground_pts_m must both be shape (4, 2).")
    h, _ = cv2.findHomography(img_pts.astype(np.float32), ground_pts_m.astype(np.float32), method=0)
    if h is None:
        raise RuntimeError("Homography solve failed.")
    return h


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sand unevenness real-time pipeline")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--roi", type=int, nargs=4, default=[0, 80, 640, 320], metavar=("X", "Y", "W", "H"))
    p.add_argument("--sandbox-w", type=float, default=1.2)
    p.add_argument("--sandbox-h", type=float, default=0.8)
    p.add_argument("--grid-w", type=int, default=20)
    p.add_argument("--grid-h", type=int, default=20)
    p.add_argument("--hfile", type=str, default="", help="Optional .npy homography file (img->robot)")
    p.add_argument("--list-cameras", action="store_true", help="Probe and print available camera indices, then exit")
    p.add_argument("--max-camera-index", type=int, default=8, help="Max index to probe with --list-cameras")
    p.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "dshow", "msmf"],
        help="VideoCapture backend (Windows: dshow/msmf).",
    )
    p.add_argument("--no-gui", action="store_true")
    return p.parse_args()


def load_or_default_h(hfile: str) -> np.ndarray:
    if hfile:
        h = np.load(hfile)
        if h.shape != (3, 3):
            raise ValueError("Homography file must contain a (3,3) matrix.")
        return h.astype(np.float32)
    return np.eye(3, dtype=np.float32)


def colorize_grid(grid: np.ndarray, scale: int = 16) -> np.ndarray:
    g = cv2.normalize(grid, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    c = cv2.applyColorMap(g, cv2.COLORMAP_TURBO)
    return cv2.resize(c, (g.shape[1] * scale, g.shape[0] * scale), interpolation=cv2.INTER_NEAREST)


def backend_flag(name: str) -> int:
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def list_cameras(max_index: int, backend_name: str) -> List[int]:
    found = []
    b = backend_flag(backend_name)
    for idx in range(max_index + 1):
        print(f"Checking camera index {idx}...", flush=True)
        cap = cv2.VideoCapture(idx, b)
        if not cap.isOpened():
            cap.release()
            print("  -> not opened", flush=True)
            continue
        ok, _ = cap.read()
        cap.release()
        if ok:
            print("  -> OK", flush=True)
            found.append(idx)
        else:
            print("  -> opened, but no frame", flush=True)
    return found


def main() -> None:
    args = parse_args()

    if args.list_cameras:
        print(f"Probing camera indices 0..{args.max_camera_index}")
        print(f"Backend: {args.backend}", flush=True)
        cams = list_cameras(args.max_camera_index, args.backend)
        if cams:
            print("Detected camera indices:", ", ".join(str(c) for c in cams))
            print("Use one with: --camera <index>")
        else:
            print("No cameras detected. Check USB cable, permissions, and if another app is using the camera.")
        return

    h = load_or_default_h(args.hfile)

    cam_cfg = CameraConfig(
        device_id=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        roi=tuple(args.roi),
        backend=args.backend,
    )
    grid_cfg = GridConfig(
        sandbox_w_m=args.sandbox_w,
        sandbox_h_m=args.sandbox_h,
        grid_w=args.grid_w,
        grid_h=args.grid_h,
    )

    print(f"Starting pipeline with camera index {args.camera}", flush=True)
    print(f"Frame size {args.width}x{args.height}, ROI={tuple(args.roi)}")
    try:
        pipe = SandUnevennessPipeline(
            cam_cfg=cam_cfg,
            prep_cfg=PreprocessConfig(),
            un_cfg=UnevennessConfig(),
            grid_cfg=grid_cfg,
            imu_cfg=IMUConfig(),
            img_to_robot_h=h,
        )
    except Exception as e:
        print(f"Camera startup failed: {e}")
        print("Tip: run with --list-cameras to find a working index.")
        return

    print("Running pipeline. Press ESC to exit.")
    print("Using placeholder pose/IMU stream in this demo loop.")

    t_last = time.time()
    try:
        while True:
            # Replace with your real odom + IMU feed.
            pose_xy_yaw = (grid_cfg.sandbox_w_m / 2.0, grid_cfg.sandbox_h_m / 2.0, 0.0)
            imu_roll_pitch_vib = (0.0, 0.0, 0.2)

            out = pipe.process_one_frame(pose_xy_yaw, imu_roll_pitch_vib)
            now = time.time()
            fps = 1.0 / max(now - t_last, 1e-6)
            t_last = now

            print(
                f"fps={fps:5.1f} rough={out['roughness_score']:8.2f} "
                f"sev={out['terrain_severity']:8.2f} targets={len(out['revisit_cells']):3d} "
                f"valid_proj={pipe.state.last_valid_projection_count:4d} "
                f"upd_cells={pipe.state.last_updated_cells:3d} "
                f"nonzero={int(np.count_nonzero(pipe.grid_score > 1e-4)):3d}",
                end="\r",
                flush=True,
            )

            if not args.no_gui:
                grad = out["gradient_map"]
                grad_u8 = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                grid_viz = colorize_grid(out["grid_score"])
                cv2.imshow("raw", out["frame"])
                cv2.imshow("preprocessed", out["preprocessed"])
                cv2.imshow("gradient", grad_u8)
                cv2.imshow("grid_score", grid_viz)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
            else:
                time.sleep(0.001)
    finally:
        pipe.close()
        cv2.destroyAllWindows()
        print("\nStopped.")


if __name__ == "__main__":
    main()
