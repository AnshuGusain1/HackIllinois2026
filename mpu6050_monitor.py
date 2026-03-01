#!/usr/bin/env python3
"""
MPU6050 live reader + live plot in ONE program:
- Reads MPU6050 over I2C (smbus/smbus2)
- 3D orientation box (roll/pitch)
- 2D scrolling plot for accel (m/s^2) in X,Y,Z
- Servo follows pitch (filtered) on a Raspberry Pi GPIO pin

Notes:
- This uses a complementary filter (gyro + accel) for smooth roll/pitch.
- If you only want accel-based angles, set ALPHA = 0.0.
- Yaw from MPU6050 will drift without magnetometer, so we do not use yaw in 3D box.
"""

import time
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import smbus2 as smbus
except ImportError:
    import smbus  # type: ignore

# -----------------------
# CONFIG
# -----------------------
BUS_ID = 1
MPU6050_ADDR = 0x68

HISTORY_SIZE = 200          # samples shown in 2D plot
UPDATE_INTERVAL_MS = 20     # plot update interval (ms)
TARGET_HZ = 1_000 / UPDATE_INTERVAL_MS

# MPU6050 full-scale assumptions (match what we write in init)
ACCEL_LSB_PER_G = 16384.0   # +-2g
GYRO_LSB_PER_DPS = 131.0    # +-250 dps

GRAVITY = 9.80665

# Complementary filter alpha:
# closer to 1.0 = trust gyro more (smoother), closer to 0 = trust accel more
ALPHA = 0.98

# Gyro calibration
CAL_SECONDS = 2.0           # keep still at startup
CAL_MIN_SAMPLES = 80

# Plot ranges (m/s^2)
Y_PADDING = 2.0

# -----------------------
# SERVO CONFIG (NEW)
# -----------------------
# Use BCM pin numbering (GPIO numbers).
SERVO_GPIO = 18  # GPIO18 is a great default (hardware PWM capable), but pigpio works on most pins.

# Typical servo pulse range. Many servos: 500-2500 us, some prefer 1000-2000 us.
SERVO_MIN_US = 600
SERVO_MAX_US = 2400
SERVO_CENTER_US = 1500

# Map pitch degrees to servo range.
# Example: -30 deg -> min, +30 deg -> max.
PITCH_MIN_DEG = -30.0
PITCH_MAX_DEG = +30.0

# Reduce chatter:
SERVO_DEADBAND_US = 4        # do nothing if change is smaller than this
SERVO_SMOOTH_ALPHA = 0.25    # 0..1, higher follows pitch faster

# Invert servo direction if it moves opposite of what you want.
SERVO_INVERT = False

# -----------------------
# MPU6050 REGISTERS
# -----------------------
REG_PWR_MGMT_1   = 0x6B
REG_SMPLRT_DIV   = 0x19
REG_CONFIG       = 0x1A
REG_GYRO_CONFIG  = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_ACCEL_XOUT_H = 0x3B

G = 9.80665


@dataclass
class Calibration:
    gyro_bias_dps: np.ndarray
    roll0_deg: float
    pitch0_deg: float


class MPU6050:
    def __init__(self, bus_id: int = 1, address: int = MPU6050_ADDR):
        self.bus = smbus.SMBus(bus_id)
        self.address = address
        self.accel_lsb_per_g = 16384.0  # +-2g
        self.gyro_lsb_per_dps = 131.0   # +-250 dps

    def initialize(self) -> None:
        self._write(REG_PWR_MGMT_1, 0x00)
        time.sleep(0.1)

        # DLPF config
        self._write(REG_CONFIG, 0x03)

        # Sample rate divider: sample_rate = 1kHz/(1+div)
        self._write(REG_SMPLRT_DIV, 4)

        # Accel full scale: 0x00 = +-2g
        self._write(REG_ACCEL_CONFIG, 0x00)
        self._write(REG_GYRO_CONFIG, 0x00)
        time.sleep(0.05)

    def close(self) -> None:
        self.bus.close()

    def read_accel_gyro(self) -> Tuple[np.ndarray, np.ndarray]:
        data = self.bus.read_i2c_block_data(self.address, REG_ACCEL_XOUT_H, 14)
        ax = self._to_i16(data[0], data[1])
        ay = self._to_i16(data[2], data[3])
        az = self._to_i16(data[4], data[5])
        gx = self._to_i16(data[8], data[9])
        gy = self._to_i16(data[10], data[11])
        gz = self._to_i16(data[12], data[13])

        accel_g = np.array([ax, ay, az], dtype=np.float64) / self.accel_lsb_per_g
        gyro_dps = np.array([gx, gy, gz], dtype=np.float64) / self.gyro_lsb_per_dps
        return accel_g, gyro_dps

    def _write(self, reg: int, value: int) -> None:
        self.bus.write_byte_data(self.address, reg, value)

    @staticmethod
    def _to_i16(msb: int, lsb: int) -> int:
        v = (msb << 8) | lsb
        return v - 65536 if v > 32767 else v


def accel_to_roll_pitch_rad(accel_g: np.ndarray) -> tuple[float, float]:
    ax, ay, az = accel_g
    roll = math.degrees(math.atan2(ay, az))
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
    return roll, pitch


def get_rotation_matrix(pitch: float, roll: float) -> np.ndarray:
    Rx = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,               cp * cr],
        ],
        dtype=np.float64,
    )


def calibrate_sensor(dev: MPU6050, cal_seconds: float, hz: float) -> Calibration:
    samples = max(50, int(cal_seconds * hz))
    dt = 1.0 / hz
    gyro_hist = []
    accel_hist = []

    print("Calibrating gyro bias, keep the IMU still...")
    for _ in range(samples):
        _, gyro_dps = dev.read_accel_gyro()
        gyro_hist.append(gyro_dps)
        accel_hist.append(accel_g)
        if (i + 1) % max(1, samples // 10) == 0:
            pct = int((i + 1) * 100 / samples)
            print(f"  {pct}%")
        time.sleep(dt)

    gyro_bias = np.mean(np.stack(gyro_hist, axis=0), axis=0)
    accel_mean = np.mean(np.stack(accel_hist, axis=0), axis=0)
    roll0, pitch0 = accel_to_roll_pitch_deg(accel_mean)

    return Calibration(gyro_bias_dps=gyro_bias, roll0_deg=roll0, pitch0_deg=pitch0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MPU-6050 live monitor with live XYZ accel plot")
    p.add_argument("--bus", type=int, default=1, help="I2C bus id (Pi usually 1)")
    p.add_argument("--addr", type=lambda x: int(x, 0), default=MPU6050_ADDR, help="I2C address (default 0x68)")
    p.add_argument("--hz", type=float, default=60.0, help="Update rate (Hz)")
    p.add_argument("--cal-seconds", type=float, default=3.0, help="Startup calibration duration (robot must be still)")
    p.add_argument("--alpha", type=float, default=0.98, help="Complementary filter alpha for roll/pitch")
    p.add_argument("--accel-deadband", type=float, default=0.12, help="Linear accel deadband (m/s^2)")
    p.add_argument("--vel-damping", type=float, default=0.35, help="Velocity damping per second (helps drift)")

    # Live plot options (matplotlib)
    p.add_argument("--plot", action="store_true", help="Show live matplotlib plot (needs Pi desktop/X11)")
    p.add_argument("--plot-seconds", type=float, default=12.0, help="History window for plot")
    p.add_argument("--plot-decimate", type=int, default=1, help="Update plot every N samples")
    return p.parse_args()


# -----------------------
# SERVO HELPERS (NEW)
# -----------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def pitch_deg_to_pulse_us(pitch_deg: float) -> int:
    # Clamp to your desired pitch range
    p = clamp(pitch_deg, PITCH_MIN_DEG, PITCH_MAX_DEG)

    # Normalize to 0..1
    t = (p - PITCH_MIN_DEG) / (PITCH_MAX_DEG - PITCH_MIN_DEG)

    if SERVO_INVERT:
        t = 1.0 - t

    us = SERVO_MIN_US + t * (SERVO_MAX_US - SERVO_MIN_US)
    return int(round(us))


def main() -> None:
    # --- Servo setup (NEW) ---
    try:
        import pigpio  # type: ignore
    except Exception as e:
        print("pigpio not available. Install and start pigpiod:")
        print("  sudo apt-get install -y pigpio python3-pigpio")
        print("  sudo systemctl enable pigpiod && sudo systemctl start pigpiod")
        print(f"Import error: {e}")
        return

    pi = pigpio.pi()
    if not pi.connected:
        print("Could not connect to pigpiod. Start it with:")
        print("  sudo systemctl start pigpiod")
        return

    # Set initial servo position
    servo_us = SERVO_CENTER_US
    pi.set_servo_pulsewidth(SERVO_GPIO, servo_us)

    dev = MPU6050(BUS_ID, MPU6050_ADDR)
    try:
        dev.initialize()
    except Exception as e:
        print(f"Failed to initialize MPU6050: {e}")
        print("Check wiring and I2C. Try: i2cdetect -y 1")
        pi.set_servo_pulsewidth(SERVO_GPIO, 0)
        pi.stop()
        return

    try:
        cal = calibrate_sensor(dev, cal_seconds=max(1.0, float(args.cal_seconds)), hz=hz)
    except Exception as e:
        print(f"Calibration failed: {e}")
        dev.close()
        pi.set_servo_pulsewidth(SERVO_GPIO, 0)
        pi.stop()
        return

    # Filter states from startup orientation.
    roll_deg = cal.roll0_deg
    pitch_deg = cal.pitch0_deg
    yaw_deg = 0.0
    velocity_world = np.zeros(3, dtype=np.float64)

    print()
    print("Calibration complete.")
    print(f"Gyro bias (dps): x={cal.gyro_bias_dps[0]:+.3f} y={cal.gyro_bias_dps[1]:+.3f} z={cal.gyro_bias_dps[2]:+.3f}")
    print(f"Mount zero (deg): roll0={cal.roll0_deg:+.2f} pitch0={cal.pitch0_deg:+.2f}")
    print("Running... Press Ctrl+C to stop.")

    # History buffers
    max_points = max(60, int(max(2.0, float(args.plot_seconds)) * hz))
    t_hist = deque(maxlen=max_points)
    accel_hist = deque(maxlen=max_points)

    # Matplotlib live plot setup
    plot_enabled = bool(args.plot)
    if plot_enabled:
        if plt is None:
            print("matplotlib is not installed. Install with: sudo apt install python3-matplotlib")
            plot_enabled = False
        elif os.environ.get("DISPLAY", "") == "":
            print("No DISPLAY detected (likely SSH/headless). Live matplotlib window cannot open.")
            plot_enabled = False

    if plot_enabled:
        plt.ion()
        fig, ax = plt.subplots()
        line_ax, = ax.plot([], [], label="ax")
        line_ay, = ax.plot([], [], label="ay")
        line_az, = ax.plot([], [], label="az")
        ax.set_title("Live Linear Acceleration (m/s^2), autoscaling")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("m/s^2")
        ax.legend(loc="upper right")
        fig.tight_layout()

    last_t = time.perf_counter()

    # -----------------------
    # Matplotlib setup
    # -----------------------
    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(wspace=0.3)

    ax_3d = fig.add_subplot(121, projection="3d")
    ax_2d = fig.add_subplot(122)

    ax_3d.set_xlim([-2, 2])
    ax_3d.set_ylim([-2, 2])
    ax_3d.set_zlim([-2, 2])
    ax_3d.set_title("3D Orientation (roll/pitch)")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")

    base_box = np.array(
        [
            [-1, -0.5, -0.1],
            [1, -0.5, -0.1],
            [1, 0.5, -0.1],
            [-1, 0.5, -0.1],
            [-1, -0.5, 0.1],
            [1, -0.5, 0.1],
            [1, 0.5, 0.1],
            [-1, 0.5, 0.1],
        ],
        dtype=np.float64,
    )

    init_faces = [
        [base_box[j] for j in [0, 1, 2, 3]],
        [base_box[j] for j in [4, 5, 6, 7]],
        [base_box[j] for j in [0, 1, 5, 4]],
        [base_box[j] for j in [2, 3, 7, 6]],
        [base_box[j] for j in [1, 2, 6, 5]],
        [base_box[j] for j in [4, 7, 3, 0]],
    ]
    poly = art3d.Poly3DCollection(init_faces, facecolors="cyan", linewidths=1, edgecolors="r", alpha=0.5)
    ax_3d.add_collection3d(poly)

    x_line, = ax_2d.plot(np.arange(HISTORY_SIZE), list(accel_x), label="X", color="r")
    y_line, = ax_2d.plot(np.arange(HISTORY_SIZE), list(accel_y), label="Y", color="g")
    z_line, = ax_2d.plot(np.arange(HISTORY_SIZE), list(accel_z), label="Z", color="b")

    ax_2d.set_title("Acceleration (m/s^2)")
    ax_2d.set_xlabel("Samples")
    ax_2d.set_ylabel("m/s^2")
    ax_2d.legend(loc="upper right")

    angle_text = ax_3d.text2D(0.02, 0.95, "", transform=ax_3d.transAxes)

    def update(_frame: int):
        nonlocal roll_f, pitch_f, last_t, servo_us

        now = time.perf_counter()
        dt = max(1e-4, now - last_t)
        last_t = now

        try:
            accel_g, gyro_dps = dev.read_accel_gyro()
        except Exception:
            return (poly, x_line, y_line, z_line, angle_text)

        gyro_dps = gyro_dps - gyro_bias

        accel_ms2 = accel_g * GRAVITY
        accel_x.append(float(accel_ms2[0]))
        accel_y.append(float(accel_ms2[1]))
        accel_z.append(float(accel_ms2[2]))

        roll_acc, pitch_acc = accel_to_roll_pitch_rad(accel_g)

        alpha = float(np.clip(ALPHA, 0.0, 1.0))
        roll_gyro = roll_f + math.radians(float(gyro_dps[0])) * dt
        pitch_gyro = pitch_f + math.radians(float(gyro_dps[1])) * dt
        roll_f = alpha * roll_gyro + (1.0 - alpha) * roll_acc
        pitch_f = alpha * pitch_gyro + (1.0 - alpha) * pitch_acc

        # -----------------------
        # SERVO UPDATE (NEW)
        # -----------------------
        pitch_deg = math.degrees(pitch_f)
        target_us = pitch_deg_to_pulse_us(pitch_deg)

        # Smooth pulse changes to reduce jitter
        target_us_f = int(round(
            (1.0 - SERVO_SMOOTH_ALPHA) * servo_us + SERVO_SMOOTH_ALPHA * target_us
        ))

        # Deadband to prevent constant tiny updates
        if abs(target_us_f - servo_us) >= SERVO_DEADBAND_US:
            servo_us = target_us_f
            pi.set_servo_pulsewidth(SERVO_GPIO, servo_us)

        # Update 3D box
        R = get_rotation_matrix(pitch_f, roll_f)
        rotated_box = base_box @ R.T
        faces = [
            [rotated_box[j] for j in [0, 1, 2, 3]],
            [rotated_box[j] for j in [4, 5, 6, 7]],
            [rotated_box[j] for j in [0, 1, 5, 4]],
            [rotated_box[j] for j in [2, 3, 7, 6]],
            [rotated_box[j] for j in [1, 2, 6, 5]],
            [rotated_box[j] for j in [4, 7, 3, 0]],
        ]
        poly.set_verts(faces)

        angle_text.set_text(
            f"roll={math.degrees(roll_f):+.1f}°\n"
            f"pitch={pitch_deg:+.1f}°\n"
            f"servo={servo_us} us"
        )

        x_line.set_ydata(list(accel_x))
        y_line.set_ydata(list(accel_y))
        z_line.set_ydata(list(accel_z))

        all_data = list(accel_x) + list(accel_y) + list(accel_z)
        ymin = min(all_data) - Y_PADDING
        ymax = max(all_data) + Y_PADDING
        ax_2d.set_ylim(ymin, ymax)

        return (poly, x_line, y_line, z_line, angle_text)

    ani = FuncAnimation(fig, update, interval=UPDATE_INTERVAL_MS, blit=False, cache_frame_data=False)

    try:
        while True:
            now = time.perf_counter()
            dt = max(1e-4, now - last_t)
            last_t = now

            accel_g, gyro_dps = dev.read_accel_gyro()
            gyro_dps = gyro_dps - cal.gyro_bias_dps

            # Complementary filter for roll/pitch
            acc_roll_deg, acc_pitch_deg = accel_to_roll_pitch_deg(accel_g)
            alpha = float(np.clip(args.alpha, 0.0, 1.0))
            roll_deg = alpha * (roll_deg + gyro_dps[0] * dt) + (1.0 - alpha) * acc_roll_deg
            pitch_deg = alpha * (pitch_deg + gyro_dps[1] * dt) + (1.0 - alpha) * acc_pitch_deg
            yaw_deg += gyro_dps[2] * dt

            # Convert to world frame and remove gravity
            accel_ms2 = accel_g * G
            R_bw = rotation_body_to_world(roll_deg, pitch_deg, yaw_deg)
            accel_world = R_bw @ accel_ms2
            linear_world = accel_world - np.array([0.0, 0.0, G], dtype=np.float64)

            # Deadband to reduce drift from noise
            deadband = max(0.0, float(args.accel_deadband))
            linear_world[np.abs(linear_world) < deadband] = 0.0

            # (Optional) velocity estimate (still drifts, but useful)
            velocity_world += linear_world * dt
            damping = max(0.0, float(args.vel_damping))
            velocity_world *= math.exp(-damping * dt)

            # Store history for plotting
            t_rel = now - t0
            t_hist.append(t_rel)
            accel_hist.append((float(linear_world[0]), float(linear_world[1]), float(linear_world[2])))

            # Terminal readout
            speed = float(np.linalg.norm(velocity_world))
            lin_mag = float(np.linalg.norm(linear_world))

            rel_roll = roll_deg - cal.roll0_deg
            rel_pitch = pitch_deg - cal.pitch0_deg
            rel_yaw = yaw_deg

            print("\x1b[2J\x1b[H", end="")
            print("MPU-6050 Live Monitor")
            print(f"Angle (deg)      roll={rel_roll:+7.2f}  pitch={rel_pitch:+7.2f}  yaw={rel_yaw:+7.2f}")
            print(
                "Velocity (m/s)   "
                f"vx={velocity_world[0]:+7.3f}  vy={velocity_world[1]:+7.3f}  vz={velocity_world[2]:+7.3f}  |v|={speed:6.3f}"
            )
            print(
                "Linear Accel     "
                f"ax={linear_world[0]:+7.3f}  ay={linear_world[1]:+7.3f}  az={linear_world[2]:+7.3f}  |a|={lin_mag:6.3f}  m/s^2"
            )
            print(
                "Raw Accel (g)    "
                f"ax={accel_g[0]:+6.3f}  ay={accel_g[1]:+6.3f}  az={accel_g[2]:+6.3f}"
            )
            print(
                "Gyro (dps)       "
                f"gx={gyro_dps[0]:+7.3f}  gy={gyro_dps[1]:+7.3f}  gz={gyro_dps[2]:+7.3f}"
            )
            print("Ctrl+C to stop")

            # Live plot (autoscaling)
            sample_i += 1
            if plot_enabled and (sample_i % max(1, int(args.plot_decimate)) == 0):
                t_np = np.array(t_hist, dtype=np.float64)
                acc_np = np.array(accel_hist, dtype=np.float64)

                if t_np.size >= 2:
                    window_s = max(2.0, float(args.plot_seconds))
                    x0 = max(0.0, t_np[-1] - window_s)
                    keep = t_np >= x0

                    t_view = t_np[keep]
                    a_view = acc_np[keep]  # columns: ax, ay, az

                    line_ax.set_data(t_view, a_view[:, 0])
                    line_ay.set_data(t_view, a_view[:, 1])
                    line_az.set_data(t_view, a_view[:, 2])

                    ax.relim()
                    ax.autoscale_view()

                    plt.draw()
                    plt.pause(0.001)

            elapsed = time.perf_counter() - now
            time.sleep(max(0.0, dt_target - elapsed))

    except KeyboardInterrupt:
        pass
    finally:
        # Stop sending pulses so the servo relaxes
        pi.set_servo_pulsewidth(SERVO_GPIO, 0)
        pi.stop()
        dev.close()
        if plot_enabled and plt is not None:
            try:
                plt.ioff()
                plt.close("all")
            except Exception:
                pass
        print("\nStopped.")


if __name__ == "__main__":
    main()