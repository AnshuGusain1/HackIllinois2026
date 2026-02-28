#!/usr/bin/env python3
"""
demo.py
~~~~~~~
End-to-end demonstration of the sand-leveling robot planner.

Scenario:
  - 3 m x 2 m sandbox
  - One rectangular obstacle (0.3 x 0.3 m) placed off-center
  - Robot with 15 cm radius, 5 cm clearance, 30 cm tool width
  - Synthetic unevenness map with a few hot cells for refinement
  - Plots coverage path, runs kinematic sim, prints sanity report
"""

import math
import random
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")   # headless backend for server environments
import matplotlib.pyplot as plt

from sand_robot import (
    Pose, Waypoint, SandboxConfig, ObstacleConfig, RobotConfig,
    PlannerParams, ToolMode, UnevennessMap,
    build_coverage_path, build_refinement_path, build_keepout,
    estimate_path_length, estimate_run_time,
    PurePursuitController, IMUReading, ToolStateMachine,
    suggest_relocalization,
    plot_path, plot_unevenness_map, run_kinematic_sim, sanity_report,
)


def make_synthetic_unevenness(
    sandbox: SandboxConfig,
    rows: int = 20,
    cols: int = 20,
    seed: int = 42,
) -> UnevennessMap:
    """Create a synthetic unevenness map with a few hot spots."""
    rng = random.Random(seed)
    cw = sandbox.width / cols
    ch = sandbox.height / rows
    data = []
    for r in range(rows):
        row = []
        for c in range(cols):
            # Base noise
            val = rng.gauss(0.2, 0.15)
            # Add a hot spot in the upper-right quadrant
            if 12 <= c <= 16 and 14 <= r <= 18:
                val += rng.uniform(0.4, 0.8)
            # Add another near center-left
            if 3 <= c <= 5 and 8 <= r <= 10:
                val += rng.uniform(0.3, 0.6)
            row.append(max(0.0, min(1.0, val)))
        data.append(row)

    return UnevennessMap(
        rows=rows, cols=cols,
        cell_width=cw, cell_height=ch,
        data=data, revisit_threshold=0.55,
    )


def main():
    # ------------------------------------------------------------------
    # 1. Define scenario
    # ------------------------------------------------------------------
    sandbox = SandboxConfig(width=3.0, height=2.0)

    obstacle = ObstacleConfig(
        x_min=1.2, x_max=1.5,
        y_min=0.7, y_max=1.0,
    )

    robot = RobotConfig(
        footprint_radius=0.15,
        safety_clearance=0.05,
        tool_width=0.30,
        min_turn_radius=0.20,
        max_speed=0.50,
        wheelbase=0.25,
    )

    params = PlannerParams(
        line_spacing=0.25,       # 5 cm overlap with 30 cm tool
        turn_segments=10,
        detour_resolution=0.04,
        revisit_extra_passes=2,
        lookahead_dist=0.30,
        goal_tolerance=0.05,
        heading_kp=2.0,
        pitch_slow_thresh_deg=8.0,
        pitch_estop_thresh_deg=20.0,
        max_run_time_s=600.0,
    )

    print("Building coverage path...")
    keepout = build_keepout(obstacle, robot.inflation_radius)

    # ------------------------------------------------------------------
    # 2. Phase 1: full-coverage raster path
    # ------------------------------------------------------------------
    coverage_wps = build_coverage_path(sandbox, robot, obstacle, params)
    print(f"  Phase 1: {len(coverage_wps)} waypoints, "
          f"{estimate_path_length(coverage_wps):.1f} m, "
          f"{estimate_run_time(coverage_wps):.0f} s")

    # ------------------------------------------------------------------
    # 3. Phase 2: adaptive refinement
    # ------------------------------------------------------------------
    umap = make_synthetic_unevenness(sandbox)
    revisit_cells = umap.cells_needing_revisit()
    print(f"  Cells needing revisit: {len(revisit_cells)}")

    last_pose = coverage_wps[-1].pose if coverage_wps else Pose(0, 0, 0)
    refinement_wps = build_refinement_path(
        revisit_cells, umap, last_pose, params, keepout=keepout,
    )
    print(f"  Phase 2: {len(refinement_wps)} waypoints, "
          f"{estimate_path_length(refinement_wps):.1f} m, "
          f"{estimate_run_time(refinement_wps):.0f} s")

    # Merge phases
    all_wps = coverage_wps + refinement_wps
    print(f"  Combined: {len(all_wps)} waypoints")

    # ------------------------------------------------------------------
    # 4. Sanity report
    # ------------------------------------------------------------------
    report = sanity_report(all_wps, sandbox, obstacle, robot, params)
    print("\n" + report)

    # ------------------------------------------------------------------
    # 5. Run kinematic simulation of Pure Pursuit controller
    # ------------------------------------------------------------------
    print("\nRunning kinematic simulation...")
    trajectory = run_kinematic_sim(all_wps, robot, params, dt=0.05)
    print(f"  Simulated {len(trajectory)} poses")

    # ------------------------------------------------------------------
    # 6. Demonstrate tool state machine
    # ------------------------------------------------------------------
    tsm = ToolStateMachine()
    mode_changes = 0
    for wp in all_wps[:200]:
        imu = IMUReading(yaw_rad=wp.pose.yaw, vibration_rms=0.5)
        tsm.update(wp.tool_mode, unevenness=0.3, imu=imu)
        if tsm.changed:
            mode_changes += 1
    print(f"  Tool mode transitions in first 200 wps: {mode_changes}")

    # ------------------------------------------------------------------
    # 7. Relocalization check
    # ------------------------------------------------------------------
    advice = suggest_relocalization(
        Pose(0.1, 1.0, 0.0), sandbox, drift_estimate=0.03,
    )
    print(f"  Relocalization advice (near wall, 3cm drift): {advice}")

    # ------------------------------------------------------------------
    # 8. Plot results
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # (a) Coverage path
    plot_path(all_wps, sandbox, obstacle, robot, keepout,
              title="Full Coverage + Refinement Path",
              ax=axes[0], show=False)

    # (b) Unevenness map
    plot_unevenness_map(umap, sandbox, revisit_cells, ax=axes[1])

    # (c) Simulated trajectory vs planned
    axes[2].set_title("Simulated Trajectory (Pure Pursuit)")
    axes[2].plot(
        [p.x for p in trajectory], [p.y for p in trajectory],
        "-", color="#2a9d8f", linewidth=0.6, label="Actual trajectory",
    )
    axes[2].plot(
        [wp.pose.x for wp in all_wps], [wp.pose.y for wp in all_wps],
        ".", color="#e63946", markersize=1, alpha=0.3, label="Planned waypoints",
    )
    if keepout:
        x_ko, y_ko = keepout.exterior.xy
        axes[2].fill(x_ko, y_ko, alpha=0.15, color="red")
        axes[2].plot(x_ko, y_ko, "r--", linewidth=0.8)
    axes[2].set_xlim(-0.1, sandbox.width + 0.1)
    axes[2].set_ylim(-0.1, sandbox.height + 0.1)
    axes[2].set_aspect("equal")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "/home/claude/sand_robot_demo.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 9. Edge cases discussion (printed)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  EDGE CASES AND MITIGATIONS")
    print("=" * 60)
    edge_cases = [
        ("Obstacle near wall",
         "The inflated keep-out may extend beyond the sandbox boundary. "
         "Shapely's intersection with the sandbox rectangle clips it "
         "automatically. Raster lines near the wall will simply have "
         "shorter free segments. If the gap between obstacle and wall is "
         "narrower than the robot, the planner naturally skips those lines "
         "(no free segment survives clipping)."),
        ("Narrow corridor (< robot width)",
         "If the inflated keep-out leaves a corridor narrower than "
         "2 * footprint_radius, no raster line fits. The planner skips "
         "the corridor entirely (acceptable: the tool cannot physically "
         "reach there). Log a warning so the operator knows about "
         "uncovered area."),
        ("Localization drift",
         "Over 10 minutes of differential-drive odometry, drift can reach "
         "5-10 cm. Mitigations: (a) fuse IMU yaw to eliminate heading "
         "drift (the dominant error source); (b) at each sandbox-edge "
         "U-turn, run a quick wall-alignment routine using a rangefinder "
         "to reset the lateral position; (c) if AprilTags are mounted on "
         "sandbox corners, trigger a tag detection every N lines to get "
         "an absolute fix."),
        ("Obstacle covers an entire raster line",
         "All free segments for that line are empty. The planner simply "
         "generates a transit arc to the next line. Coverage loss is "
         "limited to the obstacle footprint (which cannot be covered "
         "anyway)."),
        ("Multiple obstacles",
         "Call build_keepout for each obstacle and union the polygons "
         "(Shapely's unary_union). The clip-and-detour logic works on "
         "the combined keep-out with no changes."),
        ("Run-time budget exceeded",
         "The sanity_report flags this. Options: increase line_spacing "
         "(less overlap), reduce revisit_extra_passes, switch more "
         "segments to SPREAD or SMOOTH mode (faster), or accept partial "
         "coverage and prioritize the most uneven regions first."),
    ]
    for title, body in edge_cases:
        print(f"\n  [{title}]")
        # Wrap text
        words = body.split()
        line = "    "
        for w in words:
            if len(line) + len(w) + 1 > 72:
                print(line)
                line = "    " + w
            else:
                line += " " + w if line.strip() else "    " + w
        print(line)

    print("\n" + "=" * 60)
    print("  TUNING GUIDANCE")
    print("=" * 60)
    tuning = [
        ("line_spacing",
         "Set to 80-90% of tool_width for adequate overlap. "
         "Tighter spacing improves quality but increases run time linearly."),
        ("lookahead_dist",
         "Start at 1.0-1.5x the line_spacing. Too short causes oscillation; "
         "too long cuts corners on detours."),
        ("min_turn_radius",
         "Measure empirically: command a slow spin and find the tightest "
         "radius at which the wheels do not slip in sand."),
        ("safety_clearance",
         "5 cm is reasonable for a static obstacle with good localization. "
         "Increase to 10+ cm if localization is poor."),
        ("pitch thresholds",
         "Calibrate on a known slope. 8 deg slow-down and 20 deg e-stop "
         "are conservative starting points."),
        ("revisit_extra_passes",
         "2 passes usually suffice for moderate unevenness. Increase for "
         "deeply sculpted areas; monitor run-time budget."),
    ]
    for name, guidance in tuning:
        print(f"\n  {name}:")
        words = guidance.split()
        line = "    "
        for w in words:
            if len(line) + len(w) + 1 > 72:
                print(line)
                line = "    " + w
            else:
                line += " " + w if line.strip() else "    " + w
        print(line)

    print("\nDone.")


if __name__ == "__main__":
    main()
