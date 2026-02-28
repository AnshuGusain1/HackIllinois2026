"""
sand_robot.types
~~~~~~~~~~~~~~~~
Shared dataclasses and enumerations for the autonomous sand-leveling robot.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pose:
    """2-D pose: position + heading (rad, CCW from +X)."""
    x: float
    y: float
    yaw: float = 0.0

    def dist_to(self, other: Pose) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def as_xy(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class Waypoint:
    """A waypoint along the coverage path."""
    pose: Pose
    tool_mode: ToolMode = None          # type: ignore[assignment]
    speed_mps: float = 0.3              # default CUT speed
    is_detour: bool = False             # True for obstacle-avoidance segments
    is_revisit: bool = False            # True for adaptive-refinement segments

    def __post_init__(self):
        if self.tool_mode is None:
            object.__setattr__(self, "tool_mode", ToolMode.CUT)


# ---------------------------------------------------------------------------
# Sandbox, obstacle, and robot configuration
# ---------------------------------------------------------------------------

@dataclass
class SandboxConfig:
    """Rectangular sandbox in world frame with origin at bottom-left corner."""
    width: float            # X extent (m)
    height: float           # Y extent (m)

    @property
    def long_axis(self) -> str:
        """Which axis is longer -- raster lines run along this axis."""
        return "x" if self.width >= self.height else "y"


@dataclass
class ObstacleConfig:
    """
    Static obstacle described as an axis-aligned bounding box.
    For polygonal obstacles pass a list of (x, y) vertices instead.
    """
    # Axis-aligned bounding box (simple case)
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    # Or polygon vertices (takes precedence when non-empty)
    vertices: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class RobotConfig:
    """Physical and kinematic parameters of the differential-drive robot."""
    footprint_radius: float = 0.15      # r (m)
    safety_clearance: float = 0.05      # c (m)
    tool_width: float = 0.30            # effective leveling width (m)
    min_turn_radius: float = 0.20       # tightest feasible turn (m)
    max_speed: float = 0.5              # m/s
    wheelbase: float = 0.25             # track width for diff-drive (m)

    @property
    def inflation_radius(self) -> float:
        """Keep-out inflation = footprint + clearance."""
        return self.footprint_radius + self.safety_clearance


# ---------------------------------------------------------------------------
# Planner parameters
# ---------------------------------------------------------------------------

@dataclass
class PlannerParams:
    """Tunable planning parameters (all distances in meters)."""
    # Raster
    line_spacing: float = 0.25          # < tool_width for overlap
    turn_segments: int = 8              # arc discretization for U-turns
    # Obstacle
    detour_resolution: float = 0.05     # spacing of waypoints on detour arc
    # Refinement
    revisit_extra_passes: int = 2       # how many extra passes per bad cell
    # Controller
    lookahead_dist: float = 0.30        # Pure Pursuit lookahead (m)
    goal_tolerance: float = 0.05        # within this distance -> next waypoint
    heading_kp: float = 2.0             # proportional gain on heading error
    # Dune detection
    pitch_slow_thresh_deg: float = 8.0  # pitch above this -> slow down
    pitch_estop_thresh_deg: float = 20.0
    # Time budget
    max_run_time_s: float = 600.0       # 10 minutes


# ---------------------------------------------------------------------------
# Tool state machine
# ---------------------------------------------------------------------------

class ToolMode(enum.Enum):
    """Leveling tool operating modes."""
    IDLE    = "idle"
    CUT     = "cut"         # slow, blade down -- removes material
    SPREAD  = "spread"      # medium -- redistributes material
    SMOOTH  = "smooth"      # fast, drag/roller -- finish pass
    TRANSIT = "transit"     # blade up, moving between regions


# Speed presets for each mode (m/s) -- override via PlannerParams if needed
TOOL_SPEED: dict[ToolMode, float] = {
    ToolMode.IDLE:    0.0,
    ToolMode.CUT:     0.15,
    ToolMode.SPREAD:  0.25,
    ToolMode.SMOOTH:  0.40,
    ToolMode.TRANSIT: 0.45,
}


# ---------------------------------------------------------------------------
# Perception data (from camera pipeline)
# ---------------------------------------------------------------------------

@dataclass
class UnevennessMap:
    """
    Grid-based unevenness map from the camera perception pipeline.
    Values are non-negative floats; 0 = perfectly flat.
    """
    rows: int = 20
    cols: int = 20
    cell_width: float = 0.0   # computed from sandbox dims
    cell_height: float = 0.0
    data: List[List[float]] = field(default_factory=list)

    # Cells whose unevenness exceeds a threshold -> scheduled for revisit
    revisit_threshold: float = 0.5

    def cells_needing_revisit(self) -> List[Tuple[int, int]]:
        """Return (row, col) indices of cells above the revisit threshold."""
        bad: List[Tuple[int, int]] = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.data and self.data[r][c] > self.revisit_threshold:
                    bad.append((r, c))
        return bad

    def cell_center(self, row: int, col: int) -> Tuple[float, float]:
        """World-frame (x, y) of a cell center."""
        cx = (col + 0.5) * self.cell_width
        cy = (row + 0.5) * self.cell_height
        return (cx, cy)
