"""
sand_robot.controller
~~~~~~~~~~~~~~~~~~~~~
Path-tracking controller and tool-mode state machine.

Two controller implementations are provided:

1. **Pure Pursuit** (differential drive) -- primary implementation.
   Given a list of waypoints and the current pose, it finds the
   lookahead point and returns (linear_vel, angular_vel).

2. **Pose controller** (holonomic / omnidirectional) -- notes only,
   described in comments.  Returns (vx, vy, omega).

IMU integration
---------------
- Yaw from the IMU fuses with (or replaces) odometry heading so the
  controller always has a drift-corrected heading.
- Pitch is monitored to detect the robot climbing sand dunes.  If
  pitch exceeds a threshold the speed is reduced; if it exceeds a
  hard limit the robot stops (potential tipover).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .types import (
    Pose, Waypoint, PlannerParams, ToolMode, TOOL_SPEED, RobotConfig,
)


# ===================================================================
# IMU reading (mock / struct)
# ===================================================================

@dataclass
class IMUReading:
    """Simplified IMU state used by the controller."""
    yaw_rad: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    accel_z: float = 9.81       # gravity-compensated Z accel (vibration)
    vibration_rms: float = 0.0  # running RMS of accel magnitude


# ===================================================================
# Pure Pursuit controller (differential drive)
# ===================================================================

class PurePursuitController:
    """
    Geometric Pure Pursuit tracker for a differential-drive robot.

    The algorithm:
    1. Find the lookahead point on the path at distance L ahead.
    2. Compute the signed curvature kappa = 2 * sin(alpha) / L.
    3. Convert to (v, omega) where omega = v * kappa.
    4. Clamp omega to feasible range.

    Heading comes from the IMU yaw for robustness against wheel slip.
    """

    def __init__(
        self,
        waypoints: List[Waypoint],
        params: PlannerParams,
        robot: RobotConfig,
    ):
        self.waypoints = waypoints
        self.params = params
        self.robot = robot
        self.target_idx: int = 0
        self.finished: bool = False

    def _find_lookahead(
        self, pose: Pose,
    ) -> Tuple[Optional[Tuple[float, float]], int]:
        """
        Walk forward from the current target index and find the first
        waypoint beyond the lookahead distance.  If none, return the
        last waypoint.
        """
        L = self.params.lookahead_dist
        for i in range(self.target_idx, len(self.waypoints)):
            wp = self.waypoints[i]
            d = pose.dist_to(wp.pose)
            if d >= L:
                return wp.pose.as_xy(), i
        # Path exhausted -- head for the last waypoint
        last = self.waypoints[-1]
        return last.pose.as_xy(), len(self.waypoints) - 1

    def step(
        self,
        current_pose: Pose,
        imu: IMUReading,
        dt: float = 0.05,
    ) -> Tuple[float, float]:
        """
        Compute (linear_vel, angular_vel) for one control cycle.

        Parameters
        ----------
        current_pose : Pose
            Current estimated pose (x, y, yaw).  Yaw should already
            be fused with IMU.
        imu : IMUReading
            Latest IMU sample for dune detection and vibration.
        dt : float
            Control period (not used directly but available for
            derivative terms if needed).

        Returns
        -------
        v : float   -- forward speed (m/s)
        w : float   -- angular velocity (rad/s)
        """
        if self.finished:
            return 0.0, 0.0

        # Advance target index past waypoints we have already reached
        while self.target_idx < len(self.waypoints) - 1:
            d = current_pose.dist_to(self.waypoints[self.target_idx].pose)
            if d < self.params.goal_tolerance:
                self.target_idx += 1
            else:
                break

        if self.target_idx >= len(self.waypoints) - 1:
            d = current_pose.dist_to(self.waypoints[-1].pose)
            if d < self.params.goal_tolerance:
                self.finished = True
                return 0.0, 0.0

        # Find lookahead point
        la_pt, la_idx = self._find_lookahead(current_pose)
        if la_pt is None:
            self.finished = True
            return 0.0, 0.0

        # Use IMU yaw for heading (more reliable than odometry)
        theta = imu.yaw_rad

        # Transform lookahead to robot frame
        dx = la_pt[0] - current_pose.x
        dy = la_pt[1] - current_pose.y
        local_x = dx * math.cos(theta) + dy * math.sin(theta)
        local_y = -dx * math.sin(theta) + dy * math.cos(theta)
        L_actual = math.hypot(local_x, local_y)

        if L_actual < 1e-6:
            return 0.0, 0.0

        # Signed curvature
        kappa = 2.0 * local_y / (L_actual ** 2)

        # Desired speed from the nearest waypoint's tool mode
        current_wp = self.waypoints[min(self.target_idx, len(self.waypoints) - 1)]
        v_desired = current_wp.speed_mps

        # ---- Dune detection: slow down on steep pitch ----
        if abs(imu.pitch_deg) > self.params.pitch_estop_thresh_deg:
            # Emergency: likely climbing too steep a dune
            return 0.0, 0.0
        elif abs(imu.pitch_deg) > self.params.pitch_slow_thresh_deg:
            # Proportionally reduce speed
            scale = 1.0 - (
                (abs(imu.pitch_deg) - self.params.pitch_slow_thresh_deg)
                / (self.params.pitch_estop_thresh_deg
                   - self.params.pitch_slow_thresh_deg)
            )
            v_desired *= max(scale, 0.1)

        # ---- Vibration-based slowdown (optional) ----
        # High vibration can indicate the tool is hitting hard material
        # or an undetected obstacle.  A simple threshold works:
        VIB_LIMIT = 3.0  # m/s^2 -- tune experimentally
        if imu.vibration_rms > VIB_LIMIT:
            v_desired *= 0.5

        v = min(v_desired, self.robot.max_speed)
        w = v * kappa

        # Clamp angular velocity to kinematic limits
        w_max = v / max(self.robot.min_turn_radius, 0.01)
        w = max(-w_max, min(w, w_max))

        return v, w

    def current_tool_mode(self) -> ToolMode:
        """Return the tool mode prescribed by the current waypoint."""
        idx = min(self.target_idx, len(self.waypoints) - 1)
        return self.waypoints[idx].tool_mode

    @property
    def progress(self) -> float:
        """Fraction of waypoints completed (0..1)."""
        if not self.waypoints:
            return 1.0
        return self.target_idx / len(self.waypoints)


# ===================================================================
# Holonomic (omnidirectional) pose controller -- reference notes
# ===================================================================
#
# For a holonomic base (mecanum or omni wheels), replace Pure Pursuit
# with a simple proportional pose controller:
#
#     error_x = target.x - current.x
#     error_y = target.y - current.y
#     error_yaw = wrap_angle(target.yaw - current.yaw)
#
#     vx = kp_pos * error_x        # world frame
#     vy = kp_pos * error_y
#     omega = kp_yaw * error_yaw
#
#     # Rotate (vx, vy) into robot body frame if the low-level driver
#     # expects body-frame velocities.
#
# Gains kp_pos ~ 1.5, kp_yaw ~ 2.0 are reasonable starting points.
# Add velocity saturation and (optionally) a feedforward term for
# the next waypoint direction to reduce tracking lag.
#


# ===================================================================
# Tool state machine
# ===================================================================

class ToolStateMachine:
    """
    Manages the leveling tool operating mode based on waypoint
    prescriptions, unevenness data, and IMU feedback.

    State transitions
    -----------------
    IDLE -> CUT     : mission start
    CUT  -> SPREAD  : unevenness drops below cut_to_spread_thresh
    SPREAD -> SMOOTH: unevenness drops below spread_to_smooth_thresh
    SMOOTH -> TRANSIT: segment complete, need to reposition
    TRANSIT -> CUT  : arrived at next coverage segment
    ANY  -> IDLE    : mission complete or E-stop

    The waypoint's `tool_mode` field provides the *planned* mode.  The
    state machine can override it based on real-time sensor feedback
    (e.g., if the IMU shows excessive vibration during SMOOTH mode,
    revert to CUT).
    """

    # Unevenness thresholds (from perception pipeline)
    CUT_TO_SPREAD_THRESH: float = 0.6
    SPREAD_TO_SMOOTH_THRESH: float = 0.3

    # Vibration threshold for forcing CUT mode
    VIBRATION_CUT_THRESH: float = 4.0  # m/s^2

    def __init__(self):
        self.state: ToolMode = ToolMode.IDLE
        self._prev_state: ToolMode = ToolMode.IDLE

    def update(
        self,
        planned_mode: ToolMode,
        unevenness: float,
        imu: IMUReading,
    ) -> ToolMode:
        """
        Compute the actual tool mode for this control cycle.

        Parameters
        ----------
        planned_mode : ToolMode
            Mode prescribed by the current waypoint.
        unevenness : float
            Local unevenness reading from perception (0 = flat).
        imu : IMUReading
            Current IMU state.

        Returns
        -------
        ToolMode to command.
        """
        self._prev_state = self.state

        # Respect TRANSIT / IDLE from the planner unconditionally
        if planned_mode in (ToolMode.TRANSIT, ToolMode.IDLE):
            self.state = planned_mode
            return self.state

        # Vibration override: if the tool is bouncing hard, drop to CUT
        if imu.vibration_rms > self.VIBRATION_CUT_THRESH:
            self.state = ToolMode.CUT
            return self.state

        # Sensor-based upgrade / downgrade
        if unevenness > self.CUT_TO_SPREAD_THRESH:
            self.state = ToolMode.CUT
        elif unevenness > self.SPREAD_TO_SMOOTH_THRESH:
            self.state = ToolMode.SPREAD
        else:
            self.state = ToolMode.SMOOTH

        return self.state

    @property
    def changed(self) -> bool:
        return self.state != self._prev_state


# ===================================================================
# Relocalization helper (stub)
# ===================================================================

def suggest_relocalization(
    current_pose: Pose,
    sandbox: "SandboxConfig",
    drift_estimate: float,
) -> str:
    """
    Propose a relocalization strategy based on estimated drift.

    In practice this would trigger an AprilTag scan or wall-following
    maneuver.  Here we just return an advisory string.
    """
    if drift_estimate < 0.02:
        return "DRIFT_OK"

    # If near a wall, use wall contact / range sensor for correction
    margin = 0.3
    near_wall = (
        current_pose.x < margin
        or current_pose.x > sandbox.width - margin
        or current_pose.y < margin
        or current_pose.y > sandbox.height - margin
    )
    if near_wall:
        return "RELOCALIZE_WALL_ALIGN"

    return "RELOCALIZE_APRILTAG"
