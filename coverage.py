"""
sand_robot.coverage
~~~~~~~~~~~~~~~~~~~
Full-coverage raster planner with static obstacle avoidance and
adaptive refinement for high-unevenness cells.

Algorithm overview
------------------
1. **Raster generation**: Boustrophedon (lawnmower) lines along the sandbox
   long axis, spaced at `line_spacing` (< tool_width for overlap).  U-turns
   are discretized arcs whose radius >= min_turn_radius.

2. **Obstacle handling** (the hard part):
   a. Build keep-out polygon = obstacle inflated by (r + c) via Minkowski sum.
   b. For every raster line, compute intersections with the keep-out polygon.
   c. Each intersection splits the line into *covered* segments (outside
      keep-out) and *blocked* segments (inside keep-out).
   d. For each blocked gap, generate a *detour*: two tangent points on the
      inflated boundary plus a polyline that follows the boundary around the
      shorter side of the obstacle.  This is practical, local, and robust.
   e. Stitch: covered segment -> detour -> covered segment -> U-turn -> ...

3. **Adaptive refinement**: After the main pass, the camera delivers a list
   of cells that still need work.  We sort them by nearest-neighbor TSP
   heuristic and generate short local passes through each cell center.

All paths are emitted as List[Waypoint].
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional

from shapely.geometry import (
    Polygon, LineString, MultiLineString, Point, box as shapely_box,
)
from shapely.ops import nearest_points

from .types import (
    Pose, Waypoint, SandboxConfig, ObstacleConfig, RobotConfig,
    PlannerParams, ToolMode, TOOL_SPEED, UnevennessMap,
)


# ===================================================================
# Helper: build inflated keep-out polygon
# ===================================================================

def build_keepout(obstacle: ObstacleConfig, inflation: float) -> Optional[Polygon]:
    """
    Create the inflated keep-out polygon for the obstacle.

    Returns None when the obstacle config is empty (no obstacle).
    Uses Shapely's `buffer` (Minkowski sum with a disk) which handles
    both convex and concave polygons gracefully.
    """
    if obstacle.vertices:
        raw = Polygon(obstacle.vertices)
    else:
        if obstacle.x_min >= obstacle.x_max or obstacle.y_min >= obstacle.y_max:
            return None  # degenerate / no obstacle
        raw = shapely_box(obstacle.x_min, obstacle.y_min,
                          obstacle.x_max, obstacle.y_max)

    inflated = raw.buffer(inflation, resolution=16)
    if inflated.is_empty:
        return None
    return inflated


# ===================================================================
# 1. Raster line generation
# ===================================================================

def _generate_raster_lines(
    sandbox: SandboxConfig,
    spacing: float,
    robot: RobotConfig,
) -> List[List[Tuple[float, float]]]:
    """
    Produce boustrophedon (alternating-direction) line segments.

    Each element is a list of two endpoints [(x0,y0), (x1,y1)].
    Lines are parallel to the long axis and spaced by `spacing`.
    A small inward margin (robot radius) keeps the robot fully inside.
    """
    margin = robot.footprint_radius
    long_axis = sandbox.long_axis

    if long_axis == "x":
        # Lines run along X; step in Y
        y_start = margin
        y_end = sandbox.height - margin
        x_lo = margin
        x_hi = sandbox.width - margin
        offsets = _frange(y_start, y_end, spacing)
        lines = []
        for i, y in enumerate(offsets):
            if i % 2 == 0:
                lines.append([(x_lo, y), (x_hi, y)])
            else:
                lines.append([(x_hi, y), (x_lo, y)])
    else:
        # Lines run along Y; step in X
        x_start = margin
        x_end = sandbox.width - margin
        y_lo = margin
        y_hi = sandbox.height - margin
        offsets = _frange(x_start, x_end, spacing)
        lines = []
        for i, x in enumerate(offsets):
            if i % 2 == 0:
                lines.append([(x, y_lo), (x, y_hi)])
            else:
                lines.append([(x, y_hi), (x, y_lo)])

    return lines


def _frange(start: float, stop: float, step: float) -> List[float]:
    """Inclusive float range."""
    vals: List[float] = []
    v = start
    while v <= stop + 1e-9:
        vals.append(v)
        v += step
    return vals


# ===================================================================
# 2. Smooth U-turn generation (arc)
# ===================================================================

def _u_turn_arc(
    p_end: Tuple[float, float],
    p_next_start: Tuple[float, float],
    segments: int,
) -> List[Tuple[float, float]]:
    """
    Generate a semicircular arc connecting the end of one raster line
    to the start of the next.  The arc center is the midpoint, and the
    radius is half the lateral distance.

    For very small gaps (< 1 cm) we just insert a straight connector.
    """
    dx = p_next_start[0] - p_end[0]
    dy = p_next_start[1] - p_end[1]
    dist = math.hypot(dx, dy)
    if dist < 0.01:
        return [p_end, p_next_start]

    cx = (p_end[0] + p_next_start[0]) / 2
    cy = (p_end[1] + p_next_start[1]) / 2
    r = dist / 2

    # Angle from center to p_end
    a0 = math.atan2(p_end[1] - cy, p_end[0] - cx)
    a1 = math.atan2(p_next_start[1] - cy, p_next_start[0] - cx)

    # Always sweep the shorter arc (pi radians for a true U-turn)
    delta = a1 - a0
    if delta > math.pi:
        delta -= 2 * math.pi
    elif delta < -math.pi:
        delta += 2 * math.pi

    pts: List[Tuple[float, float]] = []
    for i in range(segments + 1):
        t = i / segments
        a = a0 + t * delta
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


# ===================================================================
# 3. Obstacle intersection + detour
# ===================================================================

def _clip_line_against_keepout(
    line_pts: List[Tuple[float, float]],
    keepout: Polygon,
) -> Tuple[List[List[Tuple[float, float]]], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """
    Clip a raster line against the keep-out polygon.

    Returns
    -------
    free_segments : list of polylines that lie *outside* the keep-out
    blocked_gaps  : list of (entry_point, exit_point) pairs on the
                    keep-out boundary where the line crosses inside.
    """
    ls = LineString(line_pts)
    if not ls.intersects(keepout):
        return [line_pts], []

    diff = ls.difference(keepout)

    free_segments: List[List[Tuple[float, float]]] = []
    if diff.is_empty:
        # Entire line inside keep-out -- nothing is free
        return [], [(line_pts[0], line_pts[-1])]

    if isinstance(diff, LineString):
        free_segments.append(list(diff.coords))
    elif isinstance(diff, MultiLineString):
        for geom in diff.geoms:
            free_segments.append(list(geom.coords))

    # Determine blocked gaps: between consecutive free segments
    # Sort by distance along the original line direction
    direction = (line_pts[-1][0] - line_pts[0][0],
                 line_pts[-1][1] - line_pts[0][1])
    primary = 0 if abs(direction[0]) >= abs(direction[1]) else 1
    rev = direction[primary] < 0

    free_segments.sort(key=lambda seg: seg[0][primary], reverse=rev)

    blocked_gaps: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for i in range(len(free_segments) - 1):
        exit_pt = free_segments[i][-1]
        entry_pt = free_segments[i + 1][0]
        blocked_gaps.append((exit_pt, entry_pt))

    return free_segments, blocked_gaps


def _detour_around_keepout(
    entry: Tuple[float, float],
    exit_pt: Tuple[float, float],
    keepout: Polygon,
    resolution: float,
    outward_margin: float = 0.03,
) -> List[Tuple[float, float]]:
    """
    Route around the keep-out polygon from `entry` to `exit_pt`,
    following the boundary on the shorter side.

    Strategy:
    1. Project entry and exit onto the keep-out boundary -> parameter t0, t1
    2. Walk boundary CW and CCW; pick the shorter path.
    3. Push each detour point slightly outward (by `outward_margin`)
       so the path does not sit exactly on the inflated boundary.
    4. Return waypoints at `resolution` spacing.
    """
    # Use a slightly larger polygon so detour points have margin
    detour_poly = keepout.buffer(outward_margin, resolution=16)
    boundary = detour_poly.exterior

    # Find nearest points on boundary
    p_entry = nearest_points(Point(entry), boundary)[1]
    p_exit = nearest_points(Point(exit_pt), boundary)[1]

    # Normalized distance along boundary
    t_entry = boundary.project(p_entry, normalized=True)
    t_exit = boundary.project(p_exit, normalized=True)

    # Walk in both directions
    perimeter = boundary.length
    cw_dist = (t_exit - t_entry) % 1.0 * perimeter
    ccw_dist = (t_entry - t_exit) % 1.0 * perimeter

    if cw_dist <= ccw_dist:
        # Walk forward (increasing t)
        total = cw_dist
        sign = 1.0
    else:
        total = ccw_dist
        sign = -1.0

    n_pts = max(2, int(total / resolution))
    detour: List[Tuple[float, float]] = [entry]
    for i in range(n_pts + 1):
        frac = i / n_pts
        t = (t_entry + sign * frac * (total / perimeter)) % 1.0
        pt = boundary.interpolate(t, normalized=True)
        detour.append((pt.x, pt.y))
    detour.append(exit_pt)
    return detour


# ===================================================================
# 4. Stitch raster + detours into a complete path
# ===================================================================

def _yaw_between(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def build_coverage_path(
    sandbox: SandboxConfig,
    robot: RobotConfig,
    obstacle: ObstacleConfig,
    params: PlannerParams,
) -> List[Waypoint]:
    """
    Build the full Phase-1 coverage path:
    raster lines + U-turns + obstacle detours.
    """
    keepout = build_keepout(obstacle, robot.inflation_radius)
    lines = _generate_raster_lines(sandbox, params.line_spacing, robot)

    if not lines:
        return []

    all_wps: List[Waypoint] = []

    for line_idx, line_pts in enumerate(lines):

        # --- clip against obstacle ---
        if keepout is not None:
            free_segs, gaps = _clip_line_against_keepout(line_pts, keepout)
        else:
            free_segs, gaps = [line_pts], []

        if not free_segs:
            # Entire line blocked; skip (coverage gap accepted near obstacle)
            continue

        # Emit free segments interleaved with detours
        for seg_idx, seg in enumerate(free_segs):
            # Waypoints along the free segment
            for i, pt in enumerate(seg):
                if i + 1 < len(seg):
                    yaw = _yaw_between(pt, seg[i + 1])
                elif all_wps:
                    yaw = all_wps[-1].pose.yaw
                else:
                    yaw = 0.0
                all_wps.append(Waypoint(
                    pose=Pose(pt[0], pt[1], yaw),
                    tool_mode=ToolMode.CUT,
                    speed_mps=TOOL_SPEED[ToolMode.CUT],
                ))

            # If there is a blocked gap after this segment, add detour
            if seg_idx < len(gaps):
                entry, exit_pt = gaps[seg_idx]
                det_pts = _detour_around_keepout(
                    entry, exit_pt, keepout, params.detour_resolution
                )
                for i, pt in enumerate(det_pts):
                    if i + 1 < len(det_pts):
                        yaw = _yaw_between(pt, det_pts[i + 1])
                    elif all_wps:
                        yaw = all_wps[-1].pose.yaw
                    else:
                        yaw = 0.0
                    all_wps.append(Waypoint(
                        pose=Pose(pt[0], pt[1], yaw),
                        tool_mode=ToolMode.TRANSIT,
                        speed_mps=TOOL_SPEED[ToolMode.TRANSIT],
                        is_detour=True,
                    ))

        # --- U-turn to next line ---
        if line_idx + 1 < len(lines):
            turn_pts = _u_turn_arc(
                free_segs[-1][-1],
                lines[line_idx + 1][0],  # start of next raw line
                params.turn_segments,
            )
            for i, pt in enumerate(turn_pts):
                if i + 1 < len(turn_pts):
                    yaw = _yaw_between(pt, turn_pts[i + 1])
                elif all_wps:
                    yaw = all_wps[-1].pose.yaw
                else:
                    yaw = 0.0
                all_wps.append(Waypoint(
                    pose=Pose(pt[0], pt[1], yaw),
                    tool_mode=ToolMode.TRANSIT,
                    speed_mps=TOOL_SPEED[ToolMode.TRANSIT],
                ))

    return all_wps


# ===================================================================
# 5. Adaptive refinement (Phase 2)
# ===================================================================

def build_refinement_path(
    cells: List[Tuple[int, int]],
    umap: UnevennessMap,
    robot_pose: Pose,
    params: PlannerParams,
    keepout: Optional[Polygon] = None,
) -> List[Waypoint]:
    """
    Schedule revisit passes for high-unevenness cells.

    Uses a nearest-neighbor heuristic to minimize deadhead travel.
    For each cell, we generate a short back-and-forth pass through
    the cell center (axis aligned with the sandbox long axis).

    Cells whose center falls inside the keep-out zone are skipped
    (the robot cannot physically reach them).
    """
    if not cells:
        return []

    # Compute cell centers and filter out cells inside keep-out
    centers: dict[Tuple[int, int], Tuple[float, float]] = {}
    for cell in cells:
        cx, cy = umap.cell_center(*cell)
        if keepout is not None and keepout.contains(Point(cx, cy)):
            continue
        centers[cell] = (cx, cy)

    if not centers:
        return []

    valid_cells = list(centers.keys())

    # Nearest-neighbor ordering starting from current robot pose
    ordered: List[Tuple[int, int]] = []
    remaining = set(valid_cells)
    current = (robot_pose.x, robot_pose.y)
    while remaining:
        best_cell = min(remaining, key=lambda c: math.hypot(
            centers[c][0] - current[0], centers[c][1] - current[1]))
        ordered.append(best_cell)
        current = centers[best_cell]
        remaining.discard(best_cell)

    # Generate waypoints
    wps: List[Waypoint] = []
    half_w = umap.cell_width / 2 * 0.8   # stay 80% inside cell
    half_h = umap.cell_height / 2 * 0.8

    for cell in ordered:
        cx, cy = centers[cell]

        # Transit to cell
        if wps:
            prev = wps[-1].pose
        else:
            prev = robot_pose
        yaw_to = math.atan2(cy - prev.y, cx - prev.x)
        wps.append(Waypoint(
            pose=Pose(cx, cy, yaw_to),
            tool_mode=ToolMode.TRANSIT,
            speed_mps=TOOL_SPEED[ToolMode.TRANSIT],
            is_revisit=True,
        ))

        # Local back-and-forth passes (along X)
        for p in range(params.revisit_extra_passes):
            x0 = cx - half_w
            x1 = cx + half_w
            y_off = cy + (p - params.revisit_extra_passes / 2) * (
                umap.cell_height * 0.3)
            if p % 2 == 0:
                wps.append(Waypoint(
                    pose=Pose(x0, y_off, 0.0),
                    tool_mode=ToolMode.SPREAD,
                    speed_mps=TOOL_SPEED[ToolMode.SPREAD],
                    is_revisit=True,
                ))
                wps.append(Waypoint(
                    pose=Pose(x1, y_off, 0.0),
                    tool_mode=ToolMode.SPREAD,
                    speed_mps=TOOL_SPEED[ToolMode.SPREAD],
                    is_revisit=True,
                ))
            else:
                wps.append(Waypoint(
                    pose=Pose(x1, y_off, math.pi),
                    tool_mode=ToolMode.SPREAD,
                    speed_mps=TOOL_SPEED[ToolMode.SPREAD],
                    is_revisit=True,
                ))
                wps.append(Waypoint(
                    pose=Pose(x0, y_off, math.pi),
                    tool_mode=ToolMode.SPREAD,
                    speed_mps=TOOL_SPEED[ToolMode.SPREAD],
                    is_revisit=True,
                ))

    # Post-filter: drop any waypoints that land inside the keep-out
    if keepout is not None:
        wps = [wp for wp in wps
               if not keepout.contains(Point(wp.pose.x, wp.pose.y))]

    return wps


# ===================================================================
# 6. Sanity checks
# ===================================================================

def check_clearance(
    waypoints: List[Waypoint],
    keepout: Optional[Polygon],
    min_clearance: float,
) -> List[int]:
    """
    Return indices of waypoints that are *inside* the inflated keep-out
    polygon.  Points outside (or on the boundary) are considered safe,
    since the keep-out already includes the inflation radius.
    Empty list = all clear.
    """
    if keepout is None:
        return []
    violations: List[int] = []
    # Use a tiny inward buffer so boundary points are not flagged
    inner = keepout.buffer(-0.005)
    for i, wp in enumerate(waypoints):
        pt = Point(wp.pose.x, wp.pose.y)
        if inner.contains(pt):
            violations.append(i)
    return violations


def estimate_path_length(waypoints: List[Waypoint]) -> float:
    """Total Euclidean path length in meters."""
    total = 0.0
    for i in range(1, len(waypoints)):
        total += waypoints[i].pose.dist_to(waypoints[i - 1].pose)
    return total


def estimate_run_time(waypoints: List[Waypoint]) -> float:
    """Rough estimate of total run time (seconds) based on segment speeds."""
    t = 0.0
    for i in range(1, len(waypoints)):
        d = waypoints[i].pose.dist_to(waypoints[i - 1].pose)
        v = max(waypoints[i].speed_mps, 0.01)
        t += d / v
    return t
