import argparse
import math
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import smbus2 as smbus
except ImportError:
    import smbus  # type: ignore


MPU6050_ADDR = 0x68
REG_PWR_MGMT_1 = 0x6B
REG_SMPLRT_DIV = 0x19
REG_CONFIG = 0x1A
REG_GYRO_CONFIG = 0x1B
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
        # Wake device
        self._write(REG_PWR_MGMT_1, 0x00)
        time.sleep(0.1)
        # DLPF setting (low noise, reasonable latency)
        self._write(REG_CONFIG, 0x03)
        # Sample rate divider -> ~200 Hz when DLPF enabled (1kHz / (1+4))
        self._write(REG_SMPLRT_DIV, 4)
        # Accel range +-2g, gyro range +-250 dps
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


def accel_to_roll_pitch_deg(accel_g: np.ndarray) -> Tuple[float, float]:
    ax, ay, az = accel_g
    roll = math.degrees(math.atan2(ay, az))
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
    return roll, pitch


def rotation_body_to_world(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    # ZYX rotation: body -> world
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def calibrate_sensor(dev: MPU6050, cal_seconds: float, hz: float) -> Calibration:
    samples = max(50, int(cal_seconds * hz))
    dt = 1.0 / hz
    gyro_hist = []
    accel_hist = []

    print("Calibration: keep robot still...")
    for i in range(samples):
        accel_g, gyro_dps = dev.read_accel_gyro()
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
    p = argparse.ArgumentParser(description="MPU-6050 live monitor (angles, velocity, acceleration)")
    p.add_argument("--bus", type=int, default=1, help="I2C bus id (Pi usually 1)")
    p.add_argument("--addr", type=lambda x: int(x, 0), default=MPU6050_ADDR, help="I2C address (default 0x68)")
    p.add_argument("--hz", type=float, default=60.0, help="Output/update rate")
    p.add_argument("--cal-seconds", type=float, default=3.0, help="Startup calibration duration (robot must be still)")
    p.add_argument("--alpha", type=float, default=0.98, help="Complementary filter alpha for roll/pitch")
    p.add_argument("--accel-deadband", type=float, default=0.12, help="Linear accel deadband (m/s^2)")
    p.add_argument("--vel-damping", type=float, default=0.35, help="Velocity damping per second (helps drift)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hz = max(5.0, float(args.hz))
    dt_target = 1.0 / hz

    dev = MPU6050(bus_id=args.bus, address=args.addr)
    try:
        dev.initialize()
    except Exception as e:
        print(f"Failed to initialize MPU-6050: {e}")
        print("Check wiring and I2C (run `i2cdetect -y 1` on Raspberry Pi).")
        return

    try:
        cal = calibrate_sensor(dev, cal_seconds=max(1.0, float(args.cal_seconds)), hz=hz)
    except Exception as e:
        print(f"Calibration failed: {e}")
        dev.close()
        return

    # Initialize filter states from startup orientation.
    roll_deg = cal.roll0_deg
    pitch_deg = cal.pitch0_deg
    yaw_deg = 0.0
    velocity_world = np.zeros(3, dtype=np.float64)

    print()
    print("Calibration complete.")
    print(f"Gyro bias (dps): x={cal.gyro_bias_dps[0]:+.3f} y={cal.gyro_bias_dps[1]:+.3f} z={cal.gyro_bias_dps[2]:+.3f}")
    print(f"Mount zero (deg): roll0={cal.roll0_deg:+.2f} pitch0={cal.pitch0_deg:+.2f}")
    print("Running... Press Ctrl+C to stop.")

    last_t = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            dt = max(1e-4, now - last_t)
            last_t = now

            accel_g, gyro_dps = dev.read_accel_gyro()
            gyro_dps = gyro_dps - cal.gyro_bias_dps

            acc_roll_deg, acc_pitch_deg = accel_to_roll_pitch_deg(accel_g)
            alpha = float(np.clip(args.alpha, 0.0, 1.0))
            roll_deg = alpha * (roll_deg + gyro_dps[0] * dt) + (1.0 - alpha) * acc_roll_deg
            pitch_deg = alpha * (pitch_deg + gyro_dps[1] * dt) + (1.0 - alpha) * acc_pitch_deg
            yaw_deg += gyro_dps[2] * dt

            # Relative angles from startup mount orientation.
            rel_roll = roll_deg - cal.roll0_deg
            rel_pitch = pitch_deg - cal.pitch0_deg
            rel_yaw = yaw_deg

            accel_ms2 = accel_g * G
            R_bw = rotation_body_to_world(roll_deg, pitch_deg, yaw_deg)
            accel_world = R_bw @ accel_ms2
            linear_world = accel_world - np.array([0.0, 0.0, G], dtype=np.float64)

            # Suppress very small accel noise so integrated velocity is usable.
            deadband = max(0.0, float(args.accel_deadband))
            linear_world[np.abs(linear_world) < deadband] = 0.0

            velocity_world += linear_world * dt
            damping = max(0.0, float(args.vel_damping))
            velocity_world *= math.exp(-damping * dt)

            speed = float(np.linalg.norm(velocity_world))
            lin_mag = float(np.linalg.norm(linear_world))

            # Refresh terminal screen.
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

            elapsed = time.perf_counter() - now
            time.sleep(max(0.0, dt_target - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        dev.close()
        print("\nStopped.")


if __name__ == "__main__":
    main()
