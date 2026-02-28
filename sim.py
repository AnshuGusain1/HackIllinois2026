"""
sand_robot.sim
~~~~~~~~~~~~~~
Minimal simulation loop and matplotlib visualization for debugging
the coverage planner and tracking controller.

This is *not* a physics simulator; it is a kinematic step-through that
verifies waypoint geometry, obstacle clearance, and controller behavior.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from shapely.geometry import Polygon

from .types import (
    Pose, Waypoint, SandboxConfig, ObstacleConfig, RobotConfig,
    PlannerParams, ToolMode, TOOL_SPEED, UnevennessMap,
)
from .coverage import (
    build_coverage_path, build_refinement_path, build_keepout,
    check_clearance, estimate_path_length, estimate_run_time,
)
from .controller import PurePursuitController, IMUReading, ToolStateMachine


# ===================================================================
# Plotting helpers
# ===================================================================

_MODE_COLORS = {
    ToolMode.IDLE:    "#999999",
    ToolMode.CUT:     "#e63946",
    ToolMode.SPREAD:  "#f4a261",
    ToolMode.SMOOTH:  "#2a9d8f",
    ToolMode.TRANSIT: "#457b9d",
}


def plot_path(
    waypoints: List[Waypoint],
    sandbox: SandboxConfig,
    obstacle: ObstacleConfig,
    robot: RobotConfig,
    keepout: Optional[Polygon] = None,
    title: str = "Coverage Path",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot waypoints color-coded by tool mode, sandbox boundary,
    obstacle, and inflated keep-out zone.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.figure

    # Sandbox boundary
    ax.add_patch(mpatches.Rectangle(
        (0, 0), sandbox.width, sandbox.height,
        linewidth=2, edgecolor="black", facecolor="#fdf6e3", zorder=0,
    ))

    # Keep-out zone (inflated)
    if keepout is not None:
        x_ko, y_ko = keepout.exterior.xy
        ax.fill(x_ko, y_ko, alpha=0.2, color="red", label="Keep-out zone")
        ax.plot(x_ko, y_ko, "r--", linewidth=1)

    # Raw obstacle
    if obstacle.vertices:
        obs_poly = plt.Polygon(obstacle.vertices, closed=True,
                               facecolor="#c0392b", alpha=0.6, label="Obstacle")
        ax.add_patch(obs_poly)
    elif obstacle.x_min < obstacle.x_max:
        ax.add_patch(mpatches.Rectangle(
            (obstacle.x_min, obstacle.y_min),
            obstacle.x_max - obstacle.x_min,
            obstacle.y_max - obstacle.y_min,
            facecolor="#c0392b", alpha=0.6, label="Obstacle",
        ))

    # Path segments colored by tool mode
    for mode in ToolMode:
        xs = [wp.pose.x for wp in waypoints if wp.tool_mode == mode]
        ys = [wp.pose.y for wp in waypoints if wp.tool_mode == mode]
        if xs:
            ax.scatter(xs, ys, c=_MODE_COLORS[mode], s=4,
                       label=mode.value, zorder=2)

    # Draw path lines (thin, gray) to show connectivity
    if len(waypoints) > 1:
        xs = [wp.pose.x for wp in waypoints]
        ys = [wp.pose.y for wp in waypoints]
        ax.plot(xs, ys, "-", color="#888888", linewidth=0.4, zorder=1)

    # Start / end markers
    if waypoints:
        ax.plot(waypoints[0].pose.x, waypoints[0].pose.y,
                "g^", markersize=10, label="Start", zorder=3)
        ax.plot(waypoints[-1].pose.x, waypoints[-1].pose.y,
                "rs", markersize=10, label="End", zorder=3)

    ax.set_xlim(-0.1, sandbox.width + 0.1)
    ax.set_ylim(-0.1, sandbox.height + 0.1)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7, markerscale=2)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.tight_layout()
    return fig


def plot_unevenness_map(
    umap: UnevennessMap,
    sandbox: SandboxConfig,
    revisit_cells: List[Tuple[int, int]],
    ax: Optional[plt.Axes] = None,
) -> None:
    """Heatmap overlay of the unevenness grid with revisit cells highlighted."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))

    import numpy as np
    data = np.array(umap.data) if umap.data else np.zeros((umap.rows, umap.cols))

    ax.imshow(
        data, origin="lower", cmap="YlOrRd",
        extent=[0, sandbox.width, 0, sandbox.height],
        alpha=0.6, vmin=0, vmax=1.0,
    )
    for (r, c) in revisit_cells:
        cx, cy = umap.cell_center(r, c)
        ax.plot(cx, cy, "kx", markersize=8, markeredgewidth=2)

    ax.set_title("Unevenness Map (X = revisit cells)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")


# ===================================================================
# Kinematic simulation loop
# ===================================================================

def run_kinematic_sim(
    waypoints: List[Waypoint],
    robot: RobotConfig,
    params: PlannerParams,
    dt: float = 0.05,
    max_steps: int = 50_000,
) -> List[Pose]:
    """
    Step through the Pure Pursuit controller with ideal kinematic
    propagation (no noise) and return the actual trajectory.

    Useful for verifying the controller tracks the planned path.
    """
    if not waypoints:
        return []

    # Start at first waypoint
    pose = Pose(waypoints[0].pose.x, waypoints[0].pose.y,
                waypoints[0].pose.yaw)
    ctrl = PurePursuitController(waypoints, params, robot)
    trajectory: List[Pose] = [pose]

    for _ in range(max_steps):
        if ctrl.finished:
            break

        imu = IMUReading(yaw_rad=pose.yaw)  # ideal: IMU = true heading
        v, w = ctrl.step(pose, imu, dt)

        # Kinematic update (unicycle model)
        new_yaw = pose.yaw + w * dt
        new_x = pose.x + v * math.cos(new_yaw) * dt
        new_y = pose.y + v * math.sin(new_yaw) * dt
        pose = Pose(new_x, new_y, new_yaw)
        trajectory.append(pose)

    return trajectory


# ===================================================================
# Sanity-check report
# ===================================================================

def sanity_report(
    waypoints: List[Waypoint],
    sandbox: SandboxConfig,
    obstacle: ObstacleConfig,
    robot: RobotConfig,
    params: PlannerParams,
) -> str:
    """Print a human-readable sanity check of the planned path."""
    keepout = build_keepout(obstacle, robot.inflation_radius)

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("  SAND ROBOT -- PATH SANITY REPORT")
    lines.append("=" * 60)
    lines.append(f"Sandbox:          {sandbox.width:.2f} x {sandbox.height:.2f} m")
    lines.append(f"Line spacing:     {params.line_spacing:.3f} m "
                 f"(tool width {robot.tool_width:.3f} m, "
                 f"overlap {robot.tool_width - params.line_spacing:.3f} m)")
    lines.append(f"Inflation radius: {robot.inflation_radius:.3f} m")
    lines.append(f"Total waypoints:  {len(waypoints)}")

    path_len = estimate_path_length(waypoints)
    run_time = estimate_run_time(waypoints)
    lines.append(f"Path length:      {path_len:.1f} m")
    lines.append(f"Est. run time:    {run_time:.0f} s  ({run_time/60:.1f} min)")

    if run_time > params.max_run_time_s:
        lines.append(f"  ** WARNING: exceeds {params.max_run_time_s/60:.0f} min budget! **")

    # Clearance check
    violations = check_clearance(waypoints, keepout, robot.safety_clearance)
    if violations:
        lines.append(f"  ** CLEARANCE VIOLATIONS at {len(violations)} waypoints! **")
        for idx in violations[:5]:
            wp = waypoints[idx]
            lines.append(f"     wp[{idx}] = ({wp.pose.x:.3f}, {wp.pose.y:.3f})")
    else:
        lines.append("Clearance check:  PASS (all waypoints clear of keep-out)")

    # Boundary check
    out_of_bounds = 0
    for wp in waypoints:
        if (wp.pose.x < -0.01 or wp.pose.x > sandbox.width + 0.01
                or wp.pose.y < -0.01 or wp.pose.y > sandbox.height + 0.01):
            out_of_bounds += 1
    if out_of_bounds:
        lines.append(f"  ** {out_of_bounds} waypoints OUTSIDE sandbox bounds! **")
    else:
        lines.append("Boundary check:   PASS")

    # Mode breakdown
    mode_counts = {}
    for wp in waypoints:
        mode_counts[wp.tool_mode] = mode_counts.get(wp.tool_mode, 0) + 1
    lines.append("Waypoint modes:")
    for mode, count in sorted(mode_counts.items(), key=lambda x: x[0].value):
        lines.append(f"  {mode.value:10s}: {count:5d}")

    # Detour / revisit counts
    n_detour = sum(1 for wp in waypoints if wp.is_detour)
    n_revisit = sum(1 for wp in waypoints if wp.is_revisit)
    lines.append(f"Detour wps:       {n_detour}")
    lines.append(f"Revisit wps:      {n_revisit}")
    lines.append("=" * 60)

    return "\n".join(lines)
