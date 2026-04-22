#!/usr/bin/env python3
"""Overlay trajectories on a drone image and color by lane selection.

Expected CSV format is DataFromSky style:
- Semicolon-separated values
- Comma decimal separator
- First 8 columns are track metadata (Track ID, Type, Entry Gate, Entry Time, Exit Gate, Exit Time, Traveled Dist, Avg Speed)
- Remaining values repeat in groups of 5: x, y, speed, acc, time
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


@dataclass
class Trajectory:
    track_id: str
    entry_gate: str
    exit_gate: str
    points: list[tuple[float, float]]


def clean_text(value: str) -> str:
    return value.strip().strip('"')


def parse_decimal(value: str) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def parse_trajectory_row(row: str) -> Trajectory | None:
    fields = [field.strip() for field in row.split(";")]
    if len(fields) < 13:  # minimum: 8 metadata + 5 for one sample
        return None

    track_id = clean_text(fields[0])
    entry_gate = clean_text(fields[2])
    exit_gate = clean_text(fields[4])

    sample_values = fields[8:]
    points: list[tuple[float, float]] = []

    # DataFromSky format: 5 values per sample (x, y, speed, acc, time)
    for i in range(0, len(sample_values) - 4, 5):
        x = parse_decimal(sample_values[i])
        y = parse_decimal(sample_values[i + 1])
        if x is None or y is None:
            continue
        points.append((x, y))

    if len(points) < 2:
        return None

    return Trajectory(
        track_id=track_id,
        entry_gate=entry_gate,
        exit_gate=exit_gate,
        points=points,
    )


def load_trajectories(csv_path: Path) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    with csv_path.open("r", encoding="utf-8-sig") as f:
        _ = f.readline()  # Header
        for line in f:
            line = line.strip()
            if not line:
                continue
            trajectory = parse_trajectory_row(line)
            if trajectory is not None:
                trajectories.append(trajectory)
    return trajectories


def classify_direction(trajectory: Trajectory, inverted: bool = False) -> str:
    """Classify trajectory direction based on INITIAL heading (first portion of trajectory).
    
    This determines where the vehicle is coming FROM / their starting direction.
    Default (inverted=False): Standard image coords where top=N, bottom=S, left=W, right=E
    Inverted (inverted=True): Flipped image where top=S, bottom=N, left=E, right=W
    
    In image coords: y increases downward, x increases rightward
    """
    if len(trajectory.points) < 5:
        return "UNKNOWN"
    
    # Use initial portion of trajectory to determine starting direction
    n = len(trajectory.points)
    sample_size = max(5, n // 10)  # Use ~10% of points or at least 5
    start_points = trajectory.points[:sample_size]
    
    # Initial heading based on first segment
    dx = start_points[-1][0] - start_points[0][0]
    dy = start_points[-1][1] - start_points[0][1]
    
    # Handle near-zero movement
    if abs(dx) < 1 and abs(dy) < 1:
        return "UNCLASSIFIED"
    
    import math
    angle = math.degrees(math.atan2(dy, dx))  # -180 to 180
    
    # Classify based on angle of INITIAL heading
    # Standard: 0° = E, 90° = S, 180/-180 = W, -90° = N
    # Inverted: swap all directions (E<->W, N<->S)
    
    if -22.5 <= angle < 22.5:
        return "WB" if inverted else "EB"  # right
    elif 22.5 <= angle < 67.5:
        return "NW" if inverted else "SE"  # down-right
    elif 67.5 <= angle < 112.5:
        return "NB" if inverted else "SB"  # down
    elif 112.5 <= angle < 157.5:
        return "NE" if inverted else "SW"  # down-left
    elif angle >= 157.5 or angle < -157.5:
        return "EB" if inverted else "WB"  # left
    elif -157.5 <= angle < -112.5:
        return "SE" if inverted else "NW"  # up-left
    elif -112.5 <= angle < -67.5:
        return "SB" if inverted else "NB"  # up
    elif -67.5 <= angle < -22.5:
        return "SW" if inverted else "NE"  # up-right
    
    return "UNKNOWN"


def compute_heading(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Compute heading angle in degrees from p1 to p2."""
    import math
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-180, 180]."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def classify_turn(trajectory: Trajectory, straight_threshold: float = 25.0, inverted: bool = False) -> str:
    """Classify trajectory turn behavior using heading change.
    
    Returns: 'Straight', 'Left', 'Right', or 'UTurn'
    Left/Right is from the driver's perspective (relative to direction of travel).
    """
    import math
    
    if len(trajectory.points) < 10:
        return "Short"
    
    # Use a portion of points at start and end to compute headings (more robust than single points)
    n = len(trajectory.points)
    sample_size = max(5, n // 10)  # Use ~10% of points or at least 5
    
    # Initial heading: average over first few points
    start_points = trajectory.points[:sample_size]
    initial_heading = compute_heading(start_points[0], start_points[-1])
    
    # Final heading: average over last few points
    end_points = trajectory.points[-sample_size:]
    final_heading = compute_heading(end_points[0], end_points[-1])
    
    # Heading change (positive = clockwise on screen, negative = counter-clockwise on screen)
    heading_change = normalize_angle(final_heading - initial_heading)
    
    if abs(heading_change) < straight_threshold:
        return "Straight"
    elif abs(heading_change) > 120:
        return "UTurn"
    elif heading_change > straight_threshold:
        # Clockwise on screen (positive heading change)
        # In inverted image: clockwise = right turn from driver's POV
        # In normal image: clockwise = right turn from driver's POV
        return "Right"
    else:
        # Counter-clockwise on screen (negative heading change)
        # In inverted image: counter-clockwise = left turn from driver's POV
        # In normal image: counter-clockwise = left turn from driver's POV
        return "Left"


def classify_direction_with_turn(trajectory: Trajectory, inverted: bool = False) -> str:
    """Classify trajectory with both direction and turn behavior.
    
    Returns labels like: 'EB-Straight', 'NB-Right', 'WB-Left', etc.
    """
    direction = classify_direction(trajectory, inverted=inverted)
    turn = classify_turn(trajectory, inverted=inverted)
    
    if direction in ("UNKNOWN", "UNCLASSIFIED"):
        return direction
    
    return f"{direction}-{turn}"


def compute_turn_deviation(trajectory: Trajectory) -> float:
    """Compute maximum perpendicular deviation from straight-line path.
    
    This measures how 'tight' a turn is - smaller values = tighter turn (inner lane).
    Returns the maximum perpendicular distance from any point to the start-end line.
    """
    import math
    
    if len(trajectory.points) < 3:
        return 0.0
    
    start = trajectory.points[0]
    end = trajectory.points[-1]
    
    # Vector from start to end
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.sqrt(dx * dx + dy * dy)
    
    if length < 1:
        return 0.0
    
    # Unit normal vector (perpendicular to line)
    nx = -dy / length
    ny = dx / length
    
    # Find maximum absolute perpendicular distance
    max_deviation = 0.0
    for px, py in trajectory.points:
        # Vector from start to this point
        vx = px - start[0]
        vy = py - start[1]
        # Perpendicular distance (signed)
        perp_dist = abs(vx * nx + vy * ny)
        max_deviation = max(max_deviation, perp_dist)
    
    return max_deviation


def get_movement_base(trajectory: Trajectory, inverted: bool = False) -> tuple[str, str, str]:
    """Get direction, turn type, and turn code for a trajectory."""
    direction = classify_direction(trajectory, inverted=inverted)
    turn_type = classify_turn(trajectory, inverted=inverted)
    
    turn_codes = {
        "Left": "L",
        "Right": "R", 
        "Straight": "T",
        "UTurn": "U",
        "Short": "T",
    }
    turn_code = turn_codes.get(turn_type, "T")
    
    return direction, turn_type, turn_code


def compute_lane_thresholds(trajectories: list[Trajectory], inverted: bool = False) -> dict[str, float]:
    """Compute per-movement-type median deviation for lane classification.
    
    Returns a dict mapping movement base (e.g., 'NBL', 'WBR') to median deviation.
    """
    from collections import defaultdict
    import statistics
    
    movement_deviations: dict[str, list[float]] = defaultdict(list)
    
    for traj in trajectories:
        direction, turn_type, turn_code = get_movement_base(traj, inverted)
        
        if direction in ("UNKNOWN", "UNCLASSIFIED"):
            continue
        if turn_type in ("Straight", "Short"):
            continue  # No lane distinction for straights
            
        movement_key = f"{direction}{turn_code}"
        deviation = compute_turn_deviation(traj)
        movement_deviations[movement_key].append(deviation)
    
    # Compute median for each movement type
    thresholds = {}
    for movement_key, deviations in movement_deviations.items():
        if len(deviations) >= 2:
            thresholds[movement_key] = statistics.median(deviations)
        else:
            # Not enough data for comparison - default to inner lane
            thresholds[movement_key] = float('inf')
    
    return thresholds


def ends_in_wb_ramp(trajectory: Trajectory) -> bool:
    """Check if trajectory ends in the WB ramp exit zone (x: 3700-3860, y: 600-800)."""
    if len(trajectory.points) < 2:
        return False
    # Check last few points to ensure vehicle is exiting via ramp
    end_points = trajectory.points[-5:] if len(trajectory.points) >= 5 else trajectory.points
    for x, y in end_points:
        if 3700 <= x <= 3860 and 600 <= y <= 800:
            return True
    return False


def get_endpoint_zone(trajectory: Trajectory) -> tuple[float, float]:
    """Get the average endpoint position (last 5 points)."""
    if len(trajectory.points) < 2:
        return trajectory.points[-1]
    end_points = trajectory.points[-5:] if len(trajectory.points) >= 5 else trajectory.points
    avg_x = sum(p[0] for p in end_points) / len(end_points)
    avg_y = sum(p[1] for p in end_points) / len(end_points)
    return (avg_x, avg_y)


def get_lane_at_x(trajectory: Trajectory, target_x: float, x_tolerance: float = 100) -> float | None:
    """Get the y-coordinate where trajectory crosses a specific x value.
    
    Returns the average y of points within x_tolerance of target_x.
    """
    nearby_points = [
        (x, y) for x, y in trajectory.points 
        if abs(x - target_x) <= x_tolerance
    ]
    if not nearby_points:
        return None
    return sum(y for _, y in nearby_points) / len(nearby_points)


def classify_wbl_lane(trajectory: Trajectory) -> str:
    """Classify WB left turn lane based on starting lane position.
    
    Uses y-coordinate at x~1500 to determine which lane the vehicle started in.
    Lane 1 (inner): y ~1050-1100
    Lane 2 (outer): y > 1100
    
    Trajectories reaching x >= 2500 are reclassified as WBT (through movement).
    """
    # Check if trajectory reaches x >= 2500 (not a complete left turn)
    max_x = max(x for x, y in trajectory.points)
    if max_x >= 2500:
        return "WBT"  # Not a proper WBL - goes too far east, reclassify as through
    
    # Get y position around x=1500 where we can see the lane separation
    lane_y = get_lane_at_x(trajectory, target_x=1500, x_tolerance=200)
    
    if lane_y is None:
        # Fallback: use early points
        if len(trajectory.points) >= 10:
            early_points = trajectory.points[:10]
            lane_y = sum(y for _, y in early_points) / len(early_points)
        else:
            return "WBL1"  # default
    
    # Lane 1 is inner (lower y ~1050-1100), Lane 2 is outer (higher y)
    lane_y_threshold = 1100
    
    if lane_y < lane_y_threshold:
        return "WBL1"
    else:
        return "WBL2"


def classify_nbl_lane(trajectory: Trajectory) -> str:
    """Classify NB left turn lane based on start/pass-through position.
    
    NBL1: Starts in or passes through right lane (x: 1766-1865, y: 620-720)
    NBL2: Passes through left lane (x: 1650-1780, y: 650-720)
    
    The turn classifier already determined this is a left turn, so we just need
    to determine which lane based on the starting position.
    """
    # Check for NBL1: passes through right lane zone (x: 1766-1900, y: 620-720)
    passes_nbl1_zone = any(
        (1766 <= x <= 1885) and (655 <= y <= 679)
        for x, y in trajectory.points
    )
    if passes_nbl1_zone:
        return "NBL1"
    
    # Check for NBL2: passes through left lane zone (x: 1650-1780, y: 650-720)
    passes_nbl2_zone = any(
        (1650 <= x <= 1766) and (669 <= y <= 750)
        for x, y in trajectory.points
    )
    if passes_nbl2_zone:
        return "NBL2"
    
    # Fallback: use x position at y~680 to determine lane
    nearby_y_points = [
        (x, y) for x, y in trajectory.points 
        if 620 <= y <= 750
    ]
    if nearby_y_points:
        avg_x = sum(x for x, y in nearby_y_points) / len(nearby_y_points)
        return "NBL1" if avg_x >= 1766 else "NBL2"
    
    # Default to NBL2 if no points in expected y range
    return "NBL2"


def classify_eb_movement(trajectory: Trajectory) -> str:
    """Classify EB (Eastbound) movements based on trajectory shape.
    
    Lane boundaries defined by line segments:
    - EBR2: trajectories passing through line (2379, 846) to (2418, 732) - upper lane
    - EBR1: trajectories passing through line (2379, 846) to (2352, 955) - lower lane
    - EBR1 with end x < 1600 are actually EBT trips
    
    EBL: Left turn - Y must increase overall
    EB-Ramp: U-turn to ramp - Y decreasing, X first decreases then increases
    EBT: Through - already handled separately
    """
    if len(trajectory.points) < 5:
        return "EBT"  # Short trajectory, classify as through
    
    # Get overall trajectory stats
    start_y = trajectory.points[0][1]
    end_y = trajectory.points[-1][1]
    start_x = trajectory.points[0][0]
    end_x = trajectory.points[-1][0]
    
    y_change = end_y - start_y
    x_change = end_x - start_x
    
    # Check for X pattern (decrease then increase) for ramp
    mid_idx = len(trajectory.points) // 2
    mid_x = trajectory.points[mid_idx][0]
    x_decreases_then_increases = (mid_x < start_x) and (end_x > mid_x)
    
    # EB-Ramp: Y decreasing, X first decreases then increases
    if y_change < -50 and x_decreases_then_increases:
        return "EB-Ramp"
    
    # EBL: Left turn - Y must increase overall
    if y_change > 50:
        # Use starting lane to determine EBL1 vs EBL2
        lane_y = get_lane_at_x(trajectory, target_x=2500, x_tolerance=200)
        if lane_y is None:
            lane_y = start_y
        if lane_y < 900:
            return "EBL1"
        else:
            return "EBL2"
    
    # EBR: Right turns - Y decreases, X decreases
    if y_change < -50 and x_change < -50:
        # Classify lane based on position at x ~2400 (near boundary intersection point)
        # EBR2 line: (2379, 846) to (2418, 732) - upper lane (y < 846 at x~2400)
        # EBR1 line: (2379, 846) to (2352, 955) - lower lane (y >= 846 at x~2400)
        lane_y = get_lane_at_x(trajectory, target_x=2400, x_tolerance=150)
        if lane_y is None:
            lane_y = start_y
        
        # Check which lane boundary the trajectory is closer to
        if lane_y < 846:
            return "EBR2"  # Upper lane (through line to 732)
        else:
            # Lower lane (through line to 955)
            # But if destination x < 1600, it's actually EBT
            if end_x < 1600:
                return "EBT"
            return "EBR1"
    
    # Default: classify as through if doesn't match turning patterns
    return "EBT"


def classify_full(trajectory: Trajectory, inverted: bool = False, 
                  lane_thresholds: dict[str, float] | None = None,
                  lane_threshold: float = 150.0) -> str:
    """Classify trajectory with direction, turn, and lane.
    
    Returns labels like: 'NBL1', 'NBL2', 'EBT', 'EBR1', 'WBR', 'EB-Ramp1', 'WB-Ramp2', etc.
    Format: {Direction}{Turn}{Lane}
    - Direction: NB, SB, EB, WB (diagonal directions are excluded)
    - Turn: L (Left), R (Right), T (Through/Straight), U (UTurn -> Ramp for EB)
    - Lane: 1 (inner), 2 (outer) for left turns, U-turns/ramps, and EB right turns
    
    Right turns have only one lane except for EB which has two.
    If lane_thresholds is provided, uses per-movement-type median.
    Otherwise falls back to lane_threshold fixed value.
    """
    direction, turn_type, turn_code = get_movement_base(trajectory, inverted)
    
    if direction in ("UNKNOWN", "UNCLASSIFIED"):
        return direction
    
    # Reclassify diagonal directions to nearest cardinal direction
    if direction == "NE":
        direction = "NB"  # Reclassify as northbound
    elif direction == "NW":
        direction = "NB"  # Reclassify as northbound
    elif direction == "SE":
        direction = "SB"  # Reclassify as southbound
    elif direction == "SW":
        direction = "SB"  # Reclassify as southbound
    
    # SB trips must start in the south portion of image (high y), otherwise reclassify
    if direction == "SB":
        start_x, start_y = trajectory.points[0]
        # SB traffic enters from bottom of image (y > 1300) or mid-intersection (x: 1800-2200)
        is_valid_sb_start = (start_y > 1300) or (1800 <= start_x <= 2200 and start_y > 1100)
        if not is_valid_sb_start:
            # Reclassify based on trajectory shape
            y_change = trajectory.points[-1][1] - trajectory.points[0][1]
            x_change = trajectory.points[-1][0] - trajectory.points[0][0]
            if abs(x_change) > abs(y_change):
                direction = "EB" if x_change > 0 else "WB"
            else:
                direction = "NB"
    
    # WB vehicles passing through vertical line at x=3343 between y=829 and y=1122 are ramp trips
    if direction == "WB":
        passes_ramp_gate = any(
            3300 <= x <= 3386 and 829 <= y <= 1122
            for x, y in trajectory.points
        )
        if passes_ramp_gate:
            return "WB-Ramp2"
    
    # WB vehicles going to ramp - check ramp zone FIRST before WBL
    if direction == "WB" and ends_in_wb_ramp(trajectory):
        # Use starting lane to determine ramp lane
        lane_y = get_lane_at_x(trajectory, target_x=1500, x_tolerance=200)
        if lane_y is None and len(trajectory.points) >= 10:
            early_points = trajectory.points[:10]
            lane_y = sum(y for _, y in early_points) / len(early_points)
        lane = "1" if lane_y is not None and lane_y < 1100 else "2"
        return f"WB-Ramp{lane}"
    
    # WBL: classify by starting lane position
    if direction == "WB" and turn_type == "Left":
        return classify_wbl_lane(trajectory)
    
    # WBT: reclassify if trajectory goes below y=1062 (that's a ramp or turning trip)
    if direction == "WB" and turn_type in ("Straight", "Short"):
        min_y = min(y for x, y in trajectory.points)
        max_x = max(x for x, y in trajectory.points)
        if min_y < 1062:
            # Check if trajectory has points with y < 1000 while x is 1750-2250
            # Those are turning trips (WBL), not ramp trips
            has_turning_pattern = any(
                y < 1000 and 1750 <= x <= 2250 
                for x, y in trajectory.points
            )
            if has_turning_pattern:
                # This is a turning trip, classify as WBL
                return classify_wbl_lane(trajectory)
            # Otherwise it's a ramp trip
            return "WB-Ramp2"
        return "WBT"
    
    # WBR: check if it's actually a ramp trip (x > 3000)
    if direction == "WB" and turn_type == "Right":
        max_x = max(x for x, y in trajectory.points)
        if max_x > 3000:
            return "WB-Ramp2"  # Ramp trip, not right turn
        return "WBR"
    
    # EB trips passing through vertical line at x=3370 between y=827 and y=844 are EB-Ramp trips
    if direction == "EB":
        passes_eb_ramp_gate = any(
            3330 <= x <= 3410 and 827 <= y <= 844
            for x, y in trajectory.points
        )
        if passes_eb_ramp_gate:
            return "EB-Ramp"
    
    # EB trips passing through vertical line at x=1446 between y=841 and y=1035 are EBT trips
    if direction == "EB":
        passes_ebt_gate = any(
            1400 <= x <= 1490 and 841 <= y <= 1035
            for x, y in trajectory.points
        )
        if passes_ebt_gate:
            return "EBT"
    
    # EBT: reclassify if trajectory goes below y=750 (that's a turning trip)
    if direction == "EB" and turn_type in ("Straight", "Short"):
        min_y = min(y for x, y in trajectory.points)
        if min_y < 750:
            return classify_eb_movement(trajectory)  # Reclassify as EB turn
        return "EBT"
    
    # EB turning movements (Left, Right, UTurn) - use specialized classifier
    if direction == "EB" and turn_type in ("Left", "Right", "UTurn"):
        return classify_eb_movement(trajectory)
    
    # NBL: classify by starting lane position and destination
    if direction == "NB" and turn_type == "Left":
        return classify_nbl_lane(trajectory)
    
    # NBT: must start in north zone and pass through gate (x: 1700-2100, y: ~1320)
    if direction == "NB" and turn_type in ("Straight", "Short"):
        start_x, start_y = trajectory.points[0]
        # NB trips start at low y (top of image), around y~450-500, x~1650-1750
        valid_start = (1600 <= start_x <= 1800) and (start_y <= 550)
        passes_nbt_gate = any(
            (1700 <= x <= 2100) and (1250 <= y <= 1400)
            for x, y in trajectory.points
        )
        if valid_start and passes_nbt_gate:
            return "NBT"
        # If doesn't match NBT pattern, reclassify to UNCLASSIFIED
        return "UNCLASSIFIED"
    
    # No lane distinction for straight movements
    if turn_type in ("Straight", "Short"):
        return f"{direction}{turn_code}"
    
    # Right turns: only EB has two lanes, others have one
    if turn_type == "Right":
        return f"{direction}{turn_code}"
    
    # Get lane designation for left turns, U-turns, and EB right turns
    movement_key = f"{direction}{turn_code}"
    deviation = compute_turn_deviation(trajectory)
    
    # Use per-movement threshold if available, otherwise fixed threshold
    if lane_thresholds and movement_key in lane_thresholds:
        threshold = lane_thresholds[movement_key]
    else:
        threshold = lane_threshold
    
    # Inner lane = tighter turn = smaller deviation
    lane = "1" if deviation < threshold else "2"
    
    return f"{direction}{turn_code}{lane}"


def lane_key(trajectory: Trajectory, lane_by: str, inverted: bool = False, 
             lane_threshold: float = 150.0, lane_thresholds: dict[str, float] | None = None) -> str:
    if lane_by == "entry_gate":
        return trajectory.entry_gate or "UNKNOWN"
    if lane_by == "route":
        entry = trajectory.entry_gate or "UNKNOWN"
        exit_ = trajectory.exit_gate or "UNKNOWN"
        return f"{entry} -> {exit_}"
    if lane_by == "direction":
        return classify_direction(trajectory, inverted=inverted)
    if lane_by == "direction_turn":
        return classify_direction_with_turn(trajectory, inverted=inverted)
    if lane_by == "full":
        return classify_full(trajectory, inverted=inverted, lane_thresholds=lane_thresholds, lane_threshold=lane_threshold)
    return trajectory.exit_gate or "UNKNOWN"


# Direction-based color scheme: similar directions use similar color shades
DIRECTION_COLORS = {
    # Eastbound - Oranges
    "EB": "#ff7f0e",
    "EB-Straight": "#ff7f0e",
    "EB-Left": "#fdd0a2",
    "EB-Right": "#d94701",
    "EB-UTurn": "#fdae6b",
    "EB-Short": "#fd8d3c",
    "EBT": "#ff7f0e",
    "EBL1": "#fdd0a2",
    "EBL2": "#fdae6b",
    "EBR1": "#d94701",
    "EBR2": "#a63603",
    "EBU": "#e6550d",
    "EB-Ramp": "#ffcc00",
    "EB-Ramp1": "#ffcc00",
    "EB-Ramp2": "#ffaa00",
    # Westbound - Blues
    "WB": "#1f77b4",
    "WB-Straight": "#1f77b4",
    "WB-Left": "#6baed6",
    "WB-Right": "#08519c",
    "WB-UTurn": "#c6dbef",
    "WB-Short": "#4292c6",
    "WBT": "#1f77b4",
    "WBL1": "#6baed6",
    "WBL2": "#9ecae1",
    "WBR": "#08519c",
    "WBR1": "#08519c",
    "WBR2": "#2171b5",
    "WBU": "#4292c6",
    "WBU1": "#4292c6",
    "WBU2": "#c6dbef",
    "WB-Ramp1": "#00bfff",
    "WB-Ramp2": "#87ceeb",
    # Northbound - Reds
    "NB": "#d62728",
    "NB-Straight": "#d62728",
    "NB-Left": "#fc9272",
    "NB-Right": "#a50f15",
    "NB-UTurn": "#fcbba1",
    "NB-Short": "#fb6a4a",
    "NBT": "#d62728",
    "NBL1": "#fc9272",
    "NBL2": "#fcbba1",
    "NBR": "#a50f15",
    "NBR1": "#a50f15",
    "NBR2": "#cb181d",
    "NBU": "#fb6a4a",
    # Southbound - Greens
    "SB": "#2ca02c",
    "SB-Straight": "#2ca02c",
    "SB-Left": "#a1d99b",
    "SB-Right": "#006d2c",
    "SB-UTurn": "#c7e9c0",
    "SB-Short": "#74c476",
    "SBT": "#2ca02c",
    "SBL1": "#a1d99b",
    "SBL2": "#c7e9c0",
    "SBR": "#006d2c",
    "SBR1": "#006d2c",
    "SBR2": "#238b45",
    "SBU": "#74c476",
    # Northeast - Teals
    "NE": "#17becf",
    "NE-Straight": "#17becf",
    "NE-Left": "#9edae5",
    "NE-Right": "#0d7d8c",
    "NE-UTurn": "#c7e9f0",
    "NE-Short": "#5ab4c5",
    "NET": "#17becf",
    "NEL1": "#9edae5",
    "NEL2": "#c7e9f0",
    "NER": "#0d7d8c",
    "NER1": "#0d7d8c",
    "NER2": "#31a2b8",
    "NEU": "#5ab4c5",
    # Northwest - Purples
    "NW": "#9467bd",
    "NW-Straight": "#9467bd",
    "NW-Left": "#c5b0d5",
    "NW-Right": "#6b3d91",
    "NW-UTurn": "#dadaeb",
    "NW-Short": "#8c6bb1",
    "NWT": "#9467bd",
    "NWL1": "#c5b0d5",
    "NWL2": "#dadaeb",
    "NWR": "#6b3d91",
    "NWR1": "#6b3d91",
    "NWR2": "#7b4ea3",
    "NWU": "#8c6bb1",
    # Southeast - Pinks
    "SE": "#e377c2",
    "SE-Straight": "#e377c2",
    "SE-Left": "#f7b6d2",
    "SE-Right": "#c51b8a",
    "SE-UTurn": "#fde0ef",
    "SE-Short": "#dd6ca8",
    "SET": "#e377c2",
    "SEL1": "#f7b6d2",
    "SEL2": "#fde0ef",
    "SER": "#c51b8a",
    "SER1": "#c51b8a",
    "SER2": "#d854a6",
    "SEU": "#dd6ca8",
    # Southwest - Browns
    "SW": "#8c564b",
    "SW-Straight": "#8c564b",
    "SW-Left": "#c49c94",
    "SW-Right": "#5c3d38",
    "SW-UTurn": "#d9c2bd",
    "SW-Short": "#9c7a72",
    "SWT": "#8c564b",
    "SWL1": "#c49c94",
    "SWL2": "#d9c2bd",
    "SWR": "#5c3d38",
    "SWR1": "#5c3d38",
    "SWR2": "#745a52",
    "SWU": "#9c7a72",
    # Special
    "UNKNOWN": "#7f7f7f",
    "UNCLASSIFIED": "#bcbd22",
}


def get_lane_color(lane: str) -> str:
    """Get color for a lane, using direction-based scheme or fallback."""
    if lane in DIRECTION_COLORS:
        return DIRECTION_COLORS[lane]
    # Fallback: try to match base direction
    base = lane.split("-")[0] if "-" in lane else lane
    if base in DIRECTION_COLORS:
        return DIRECTION_COLORS[base]
    return "#7f7f7f"  # Gray fallback


def iter_points(trajectories: Iterable[Trajectory]) -> Iterable[tuple[float, float]]:
    for trajectory in trajectories:
        for point in trajectory.points:
            yield point


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay trajectories on a drone image and color by lane selection.",
    )
    parser.add_argument("--csv", default="trajectories.csv", help="Path to trajectory CSV file")
    parser.add_argument("--image", default="DroneFootage2.png", help="Path to drone frame/image used as background")
    parser.add_argument(
        "--lane-by",
        choices=["exit_gate", "entry_gate", "route", "direction", "direction_turn", "full"],
        default="full",
        help="Which attribute defines the lane color grouping (full = direction+turn+lane like NBL1)",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=0,
        help="Limit number of trajectories to draw (0 = all)",
    )
    parser.add_argument("--line-width", type=float, default=1.8, help="Trajectory line width")
    parser.add_argument("--alpha", type=float, default=0.9, help="Trajectory opacity")
    parser.add_argument("--legend-limit", type=int, default=30, help="Max legend lane entries")
    parser.add_argument("--lane-threshold", type=float, default=150.0, help="Turn deviation threshold for lane 1 vs 2 (pixels)")
    parser.add_argument("--title", default="Traffic Tracer", help="Plot title")
    parser.add_argument("--output", help="Optional output file path (PNG/JPG for image, HTML for interactive)")
    parser.add_argument(
        "--format",
        choices=["png", "html"],
        default="png",
        help="Output format: png (static image) or html (interactive)",
    )
    parser.add_argument(
        "--no-inverted",
        action="store_true",
        help="Disable inverted mode (default is inverted for image with top=S, bottom=N)",
    )
    parser.add_argument(
        "--min-y",
        type=float,
        default=0,
        help="Exclude trajectories with any point below this y value (0 = no filter)",
    )
    return parser


def get_direction_group(movement: str) -> str:
    """Extract the direction group from a movement label (e.g., NBL1 -> NB)."""
    # Handle compound directions first (NE, NW, SE, SW)
    if movement.startswith(("NE", "NW", "SE", "SW")):
        return movement[:2]
    # Then cardinal directions
    if movement.startswith(("NB", "SB", "EB", "WB")):
        return movement[:2]
    return "OTHER"


DIRECTION_GROUP_ORDER = ["NB", "SB", "EB", "WB", "NE", "NW", "SE", "SW", "OTHER"]
DIRECTION_GROUP_NAMES = {
    "NB": "Northbound",
    "SB": "Southbound",
    "EB": "Eastbound",
    "WB": "Westbound",
    "NE": "Northeast",
    "NW": "Northwest",
    "SE": "Southeast",
    "SW": "Southwest",
    "OTHER": "Other",
}


def generate_html(
    trajectories: list[Trajectory],
    lane_by: str,
    inverted: bool,
    lane_threshold: float,
    lane_thresholds: dict[str, float] | None,
    image_path: Path | None,
    title: str,
    line_width: float,
    alpha: float,
) -> str:
    """Generate an interactive HTML visualization."""
    import base64
    from collections import defaultdict, Counter
    
    # Group trajectories by movement type
    trajectories_by_lane: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        lane = lane_key(t, lane_by, inverted, lane_threshold, lane_thresholds)
        trajectories_by_lane[lane].append(t)
    
    lane_names = sorted(trajectories_by_lane.keys())
    lane_counts = {lane: len(trajs) for lane, trajs in trajectories_by_lane.items()}
    
    # Get colors
    if lane_by in ("direction", "direction_turn", "full"):
        lane_to_color = {lane: get_lane_color(lane) for lane in lane_names}
    else:
        import colorsys
        n = max(len(lane_names), 1)
        lane_to_color = {}
        for i, lane in enumerate(lane_names):
            hue = i / n
            r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            lane_to_color[lane] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    
    # Group movements by direction
    groups: dict[str, list[str]] = defaultdict(list)
    for lane in lane_names:
        group = get_direction_group(lane)
        groups[group].append(lane)
    
    # Load and encode image
    img_data = ""
    img_w, img_h = 1920, 1080  # Default dimensions
    if image_path and image_path.exists():
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_data = base64.b64encode(img_bytes).decode("utf-8")
        # Get image dimensions from header
        import struct
        def get_image_size(data: bytes):
            # PNG
            if data[:8] == b'\x89PNG\r\n\x1a\n':
                w, h = struct.unpack('>ii', data[16:24])
                return w, h
            # JPEG
            if data[:2] == b'\xff\xd8':
                i = 2
                while i < len(data) - 8:
                    if data[i] != 0xff:
                        i += 1
                        continue
                    marker = data[i+1]
                    if marker == 0xd8:  # Start
                        i += 2
                    elif marker == 0xd9:  # End
                        break
                    elif marker == 0xda:  # Start of scan
                        break
                    elif 0xc0 <= marker <= 0xc3:  # SOF markers
                        h = struct.unpack('>H', data[i+5:i+7])[0]
                        w = struct.unpack('>H', data[i+7:i+9])[0]
                        return w, h
                    else:
                        length = struct.unpack('>H', data[i+2:i+4])[0]
                        i += 2 + length
            return None, None
        w, h = get_image_size(img_bytes)
        if w and h:
            img_w, img_h = w, h
    
    # Determine image mime type
    mime_type = "image/png"
    if image_path:
        suffix = image_path.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            mime_type = "image/jpeg"
    
    # Generate SVG paths for each trajectory
    svg_paths_by_lane: dict[str, list[str]] = defaultdict(list)
    for lane, trajs in trajectories_by_lane.items():
        color = lane_to_color[lane]
        for t in trajs:
            if len(t.points) < 2:
                continue
            path_d = f"M {t.points[0][0]:.1f},{t.points[0][1]:.1f}"
            for px, py in t.points[1:]:
                path_d += f" L {px:.1f},{py:.1f}"
            svg_paths_by_lane[lane].append(
                f'<path d="{path_d}" stroke="{color}" fill="none" '
                f'stroke-width="{line_width}" stroke-opacity="{alpha}" '
                f'stroke-linecap="round" stroke-linejoin="round"/>'
            )
    
    # Build legend HTML
    legend_html = ""
    for group in DIRECTION_GROUP_ORDER:
        if group not in groups:
            continue
        movements = groups[group]
        group_name = DIRECTION_GROUP_NAMES.get(group, group)
        group_count = sum(lane_counts.get(m, 0) for m in movements)
        
        legend_html += f'''
        <div class="legend-group" data-group="{group}">
            <div class="group-header" onclick="toggleGroup('{group}')">
                <span class="group-toggle">▼</span>
                <input type="checkbox" checked onclick="event.stopPropagation(); toggleGroupVisibility('{group}', this.checked)">
                <span class="group-name">{group_name}</span>
                <span class="group-count">({group_count})</span>
            </div>
            <div class="group-items">
        '''
        for m in sorted(movements):
            color = lane_to_color[m]
            count = lane_counts.get(m, 0)
            legend_html += f'''
                <div class="legend-item" data-movement="{m}">
                    <input type="checkbox" checked onchange="toggleMovement('{m}', this.checked)">
                    <span class="color-box" style="background-color: {color};"></span>
                    <span class="movement-name">{m}</span>
                    <span class="movement-count">({count})</span>
                </div>
            '''
        legend_html += '''
            </div>
        </div>
        '''
    
    # Build SVG groups
    svg_groups = ""
    for lane in lane_names:
        paths = "\n".join(svg_paths_by_lane[lane])
        svg_groups += f'<g class="movement-group" data-movement="{lane}">\n{paths}\n</g>\n'
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            height: 100vh;
            background: #1a1a2e;
            color: #eee;
        }}
        .sidebar {{
            width: 280px;
            background: #16213e;
            padding: 16px;
            overflow-y: auto;
            border-right: 1px solid #0f3460;
        }}
        .sidebar h2 {{
            font-size: 18px;
            margin-bottom: 16px;
            color: #e94560;
        }}
        .controls {{
            margin-bottom: 16px;
            padding-bottom: 16px;
            border-bottom: 1px solid #0f3460;
        }}
        .controls button {{
            padding: 8px 12px;
            margin-right: 8px;
            margin-bottom: 8px;
            background: #0f3460;
            color: #eee;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }}
        .controls button:hover {{
            background: #e94560;
        }}
        .legend-group {{
            margin-bottom: 12px;
        }}
        .group-header {{
            display: flex;
            align-items: center;
            padding: 8px;
            background: #0f3460;
            border-radius: 4px;
            cursor: pointer;
            user-select: none;
        }}
        .group-header:hover {{
            background: #1a4a7a;
        }}
        .group-toggle {{
            width: 20px;
            font-size: 10px;
            transition: transform 0.2s;
        }}
        .group-header.collapsed .group-toggle {{
            transform: rotate(-90deg);
        }}
        .group-header input[type="checkbox"] {{
            margin-right: 8px;
        }}
        .group-name {{
            flex: 1;
            font-weight: 600;
        }}
        .group-count {{
            font-size: 12px;
            color: #888;
        }}
        .group-items {{
            padding-left: 28px;
            max-height: 500px;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }}
        .group-header.collapsed + .group-items {{
            max-height: 0;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            padding: 6px 0;
            font-size: 13px;
        }}
        .legend-item input[type="checkbox"] {{
            margin-right: 8px;
        }}
        .color-box {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
            margin-right: 8px;
        }}
        .movement-name {{
            flex: 1;
        }}
        .movement-count {{
            font-size: 11px;
            color: #888;
        }}
        .main-content {{
            flex: 1;
            overflow: auto;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .visualization-container {{
            position: relative;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }}
        .visualization-container img {{
            display: block;
            max-width: 100%;
            height: auto;
        }}
        .visualization-container svg {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }}
        .movement-group {{
            transition: opacity 0.2s;
        }}
        .movement-group.hidden {{
            opacity: 0;
            pointer-events: none;
        }}
        .stats {{
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid #0f3460;
            font-size: 12px;
            color: #888;
        }}
        .pixel-coords {{
            position: fixed;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: #00ff00;
            padding: 10px 15px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 14px;
            z-index: 1000;
            pointer-events: none;
        }}
        .pixel-coords.hidden {{
            display: none;
        }}
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>{title}</h2>
        <div class="controls">
            <button onclick="showAll()">Show All</button>
            <button onclick="hideAll()">Hide All</button>
            <button onclick="showLeftTurns()">Left Turns</button>
            <button onclick="showRightTurns()">Right Turns</button>
            <button onclick="showThroughs()">Throughs</button>
        </div>
        <div class="legend">
            {legend_html}
        </div>
        <div class="stats">
            Total trajectories: {len(trajectories)}
        </div>
    </div>
    <div class="main-content">
        <div class="visualization-container" id="vizContainer">
            <img src="data:{mime_type};base64,{img_data}" alt="Drone footage" width="{img_w}" height="{img_h}">
            <svg viewBox="0 0 {img_w} {img_h}" preserveAspectRatio="xMidYMid meet">
                {svg_groups}
            </svg>
        </div>
    </div>
    <div id="pixelCoords" class="pixel-coords hidden">X: 0, Y: 0</div>
    
    <script>
        function toggleGroup(group) {{
            const header = document.querySelector(`.legend-group[data-group="${{group}}"] .group-header`);
            header.classList.toggle('collapsed');
        }}
        
        function toggleGroupVisibility(group, visible) {{
            const items = document.querySelectorAll(`.legend-group[data-group="${{group}}"] .legend-item input[type="checkbox"]`);
            items.forEach(checkbox => {{
                checkbox.checked = visible;
                const movement = checkbox.closest('.legend-item').dataset.movement;
                toggleMovement(movement, visible);
            }});
        }}
        
        function toggleMovement(movement, visible) {{
            const group = document.querySelector(`.movement-group[data-movement="${{movement}}"]`);
            if (group) {{
                group.classList.toggle('hidden', !visible);
            }}
            updateGroupCheckbox(movement);
        }}
        
        function updateGroupCheckbox(movement) {{
            // Find which direction group this movement belongs to
            const legendItem = document.querySelector(`.legend-item[data-movement="${{movement}}"]`);
            if (!legendItem) return;
            const legendGroup = legendItem.closest('.legend-group');
            if (!legendGroup) return;
            
            const items = legendGroup.querySelectorAll('.legend-item input[type="checkbox"]');
            const allChecked = Array.from(items).every(cb => cb.checked);
            const noneChecked = Array.from(items).every(cb => !cb.checked);
            const groupCheckbox = legendGroup.querySelector('.group-header input[type="checkbox"]');
            
            groupCheckbox.checked = allChecked;
            groupCheckbox.indeterminate = !allChecked && !noneChecked;
        }}
        
        function showAll() {{
            document.querySelectorAll('.legend-item input[type="checkbox"]').forEach(cb => {{
                cb.checked = true;
                const movement = cb.closest('.legend-item').dataset.movement;
                const group = document.querySelector(`.movement-group[data-movement="${{movement}}"]`);
                if (group) group.classList.remove('hidden');
            }});
            document.querySelectorAll('.group-header input[type="checkbox"]').forEach(cb => {{
                cb.checked = true;
                cb.indeterminate = false;
            }});
        }}
        
        function hideAll() {{
            document.querySelectorAll('.legend-item input[type="checkbox"]').forEach(cb => {{
                cb.checked = false;
                const movement = cb.closest('.legend-item').dataset.movement;
                const group = document.querySelector(`.movement-group[data-movement="${{movement}}"]`);
                if (group) group.classList.add('hidden');
            }});
            document.querySelectorAll('.group-header input[type="checkbox"]').forEach(cb => {{
                cb.checked = false;
                cb.indeterminate = false;
            }});
        }}
        
        function showByPattern(pattern) {{
            hideAll();
            document.querySelectorAll('.legend-item').forEach(item => {{
                const movement = item.dataset.movement;
                if (pattern.test(movement)) {{
                    const checkbox = item.querySelector('input[type="checkbox"]');
                    checkbox.checked = true;
                    const group = document.querySelector(`.movement-group[data-movement="${{movement}}"]`);
                    if (group) group.classList.remove('hidden');
                }}
            }});
            // Update group checkboxes
            document.querySelectorAll('.legend-group').forEach(lg => {{
                const items = lg.querySelectorAll('.legend-item input[type="checkbox"]');
                const anyChecked = Array.from(items).some(cb => cb.checked);
                const allChecked = Array.from(items).every(cb => cb.checked);
                const groupCheckbox = lg.querySelector('.group-header input[type="checkbox"]');
                groupCheckbox.checked = allChecked;
                groupCheckbox.indeterminate = anyChecked && !allChecked;
            }});
        }}
        
        function showLeftTurns() {{
            showByPattern(/L[12]?$/);
        }}
        
        function showRightTurns() {{
            showByPattern(/R[12]?$/);
        }}
        
        function showThroughs() {{
            showByPattern(/T$/);
        }}
        
        // Pixel coordinate display on middle click
        const vizContainer = document.getElementById('vizContainer');
        const pixelCoords = document.getElementById('pixelCoords');
        const imgElement = vizContainer.querySelector('img');
        
        vizContainer.addEventListener('auxclick', function(e) {{
            if (e.button === 1) {{  // Middle mouse button
                e.preventDefault();
                
                const rect = imgElement.getBoundingClientRect();
                const scaleX = {img_w} / rect.width;
                const scaleY = {img_h} / rect.height;
                
                const x = Math.round((e.clientX - rect.left) * scaleX);
                const y = Math.round((e.clientY - rect.top) * scaleY);
                
                pixelCoords.textContent = `X: ${{x}}, Y: ${{y}}`;
                pixelCoords.classList.remove('hidden');
            }}
        }});
        
        // Also support regular click with Ctrl key as alternative
        vizContainer.addEventListener('click', function(e) {{
            if (e.ctrlKey) {{
                e.preventDefault();
                
                const rect = imgElement.getBoundingClientRect();
                const scaleX = {img_w} / rect.width;
                const scaleY = {img_h} / rect.height;
                
                const x = Math.round((e.clientX - rect.left) * scaleX);
                const y = Math.round((e.clientY - rect.top) * scaleY);
                
                pixelCoords.textContent = `X: ${{x}}, Y: ${{y}}`;
                pixelCoords.classList.remove('hidden');
            }}
        }});
        
        // Prevent default middle-click scroll behavior
        vizContainer.addEventListener('mousedown', function(e) {{
            if (e.button === 1) {{
                e.preventDefault();
            }}
        }});
    </script>
</body>
</html>
'''
    return html


def main() -> None:
    args = build_arg_parser().parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Auto-detect format from output extension if not explicitly set
    output_format = args.format
    if args.output:
        output_path = Path(args.output)
        if output_path.suffix.lower() == ".html":
            output_format = "html"
        elif output_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            output_format = "png"

    trajectories = load_trajectories(csv_path)
    
    # Filter by min-y (exclude trajectories with any point below this y value)
    if args.min_y > 0:
        original_count = len(trajectories)
        trajectories = [
            t for t in trajectories 
            if all(py >= args.min_y for _, py in t.points)
        ]
        print(f"Filtered by min_y={args.min_y}: {original_count} -> {len(trajectories)} trajectories")
    
    if args.max_tracks > 0:
        trajectories = trajectories[: args.max_tracks]

    if not trajectories:
        raise ValueError("No valid trajectories found in CSV.")

    inverted = not args.no_inverted  # Default is inverted=True
    lane_threshold = args.lane_threshold
    
    # For 'full' mode, compute per-movement-type lane thresholds based on median
    lane_thresholds = None
    if args.lane_by == "full":
        lane_thresholds = compute_lane_thresholds(trajectories, inverted)
        # Filter out excluded trajectories (diagonal directions)
        original_count = len(trajectories)
        trajectories = [
            t for t in trajectories 
            if lane_key(t, args.lane_by, inverted, lane_threshold, lane_thresholds) != "EXCLUDE"
        ]
        if len(trajectories) < original_count:
            print(f"Filtered diagonal directions: {original_count} -> {len(trajectories)} trajectories")
    
    lane_names = sorted({lane_key(t, args.lane_by, inverted, lane_threshold, lane_thresholds) for t in trajectories})
    
    # Print direction summary
    from collections import Counter
    lane_counts = Counter(lane_key(t, args.lane_by, inverted, lane_threshold, lane_thresholds) for t in trajectories)
    print(f"\nTrajectory counts by {args.lane_by}" + (" (inverted)" if inverted else "") + ":")
    for lane in sorted(lane_counts.keys()):
        print(f"  {lane}: {lane_counts[lane]}")
    print(f"  Total: {len(trajectories)}\n")
    
    # HTML output
    if output_format == "html":
        image_path = Path(args.image) if args.image else None
        html_content = generate_html(
            trajectories=trajectories,
            lane_by=args.lane_by,
            inverted=inverted,
            lane_threshold=lane_threshold,
            lane_thresholds=lane_thresholds,
            image_path=image_path,
            title=args.title,
            line_width=args.line_width,
            alpha=args.alpha,
        )
        
        if args.output:
            output_path = Path(args.output)
            if not output_path.suffix:
                output_path = output_path.with_suffix(".html")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html_content, encoding="utf-8")
            print(f"Saved interactive visualization to: {output_path}")
            # Open in browser
            import webbrowser
            webbrowser.open(output_path.absolute().as_uri())
        else:
            # Save to default location and open
            output_path = Path("trajectory_visualization.html")
            output_path.write_text(html_content, encoding="utf-8")
            print(f"Saved interactive visualization to: {output_path}")
            import webbrowser
            webbrowser.open(output_path.absolute().as_uri())
        return
    
    # PNG output (matplotlib)
    # Use direction-based colors for direction modes, otherwise use colormap
    if args.lane_by in ("direction", "direction_turn", "full"):
        lane_to_color = {lane: get_lane_color(lane) for lane in lane_names}
    else:
        cmap = plt.get_cmap("tab20", max(len(lane_names), 1))
        lane_to_color = {lane: cmap(i) for i, lane in enumerate(lane_names)}

    fig, ax = plt.subplots(figsize=(14, 8), dpi=120)

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = plt.imread(image_path)
        img_h, img_w = image.shape[:2]
        ax.imshow(image, origin="upper")
        ax.set_xlim(0, img_w - 1)
        ax.set_ylim(img_h - 1, 0)
    else:
        points = list(iter_points(trajectories))
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        pad = 40.0
        ax.set_facecolor("#f4f4f4")
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_max + pad, y_min - pad)

    for trajectory in trajectories:
        lane = lane_key(trajectory, args.lane_by, inverted, lane_threshold, lane_thresholds)
        color = lane_to_color[lane]
        xs = [point[0] for point in trajectory.points]
        ys = [point[1] for point in trajectory.points]
        ax.plot(xs, ys, color=color, linewidth=args.line_width, alpha=args.alpha)

    legend_items = lane_names[: max(args.legend_limit, 0)]
    handles = [
        Line2D([0], [0], color=lane_to_color[lane], lw=3, label=lane)
        for lane in legend_items
    ]
    if handles:
        ax.legend(
            handles=handles,
            title=f"Lane ({args.lane_by})",
            loc="upper right",
            fontsize=8,
            title_fontsize=9,
            frameon=True,
        )

    ax.set_title(args.title)
    ax.set_xlabel("X [px]")
    ax.set_ylabel("Y [px]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    plt.tight_layout()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
