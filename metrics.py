"""
Ego-trajectory metrics for the VLM Planning workshop challenge.

Submissions should produce a predicted ego trajectory in the local frame
(anchor at origin, forward = +X) over the 6 s / 12-step future horizon.
Use :func:`compute_ego_metrics` as the single entry point — it returns a
dictionary with every metric reported by the challenge leaderboard.

Metric summary
--------------
Displacement (local frame, metres):
    ade_2s, ade_4s, ade_6s   Mean L2 error at the 2 s / 4 s / 6 s horizons.
    fde                      L2 error at the final (6 s) step.
    miss_rate                1.0 if any step exceeds the 2 m tolerance, else 0.0.
    speed_error              Mean absolute per-step speed error (m / 0.5 s).

Heading (local frame, radians):
    ahe                      Mean absolute heading error across all steps.
    fhe                      Absolute heading error at the final step.

Map-aware (world frame, requires NuScenesMap):
    offroad                  Fraction of predicted waypoints outside the drivable area.
    offroad_rate             1.0 if any predicted waypoint is outside the drivable area, else 0.0.
    offyaw                   Mean angular deviation from the projected lane heading
                             (intersections excluded; deviations below 45 deg count as zero).

All trajectories are expected as ``np.ndarray`` of shape ``[T, 2]`` where
``T == 12`` for the future horizon.
"""

from typing import Dict

import numpy as np

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils


_OFF_YAW_THRESHOLD_RAD = np.radians(45.0)


def _projected_lane_heading(nusc_map: NuScenesMap, x: float, y: float,
                            theta_query: float = 0.0, lane_radius: float = 3.0):
    """
    Return the heading (rad) of the closest lane centreline at (x, y), or None if
    no lane is found within ``lane_radius`` metres. ``theta_query`` is used to
    disambiguate direction when projecting onto the arcline path.
    """
    lane_token = nusc_map.get_closest_lane(x, y, radius=lane_radius)
    if not lane_token:
        return None
    try:
        lane_record = nusc_map.get_arcline_path(lane_token)
        closest_pose, _ = arcline_path_utils.project_pose_to_lane((x, y, theta_query), lane_record)
        return float(closest_pose[2])
    except Exception:
        return None


def _is_in_intersection(nusc_map: NuScenesMap, x: float, y: float) -> bool:
    """True iff (x, y) falls inside a road_segment flagged as an intersection."""
    try:
        token = nusc_map.record_on_point(x, y, 'road_segment')
        if not token:
            return False
        return bool(nusc_map.get('road_segment', token).get('is_intersection', False))
    except Exception:
        return False


def _is_in_drivable_area(nusc_map: NuScenesMap, x: float, y: float) -> bool:
    """True iff (x, y) falls inside any drivable_area polygon."""
    try:
        return bool(nusc_map.record_on_point(x, y, 'drivable_area'))
    except Exception:
        return False


def _headings_from_waypoints(waypoints: np.ndarray) -> np.ndarray:
    """
    Derive per-step heading (rad) from consecutive waypoint displacements.

    A zero-origin is prepended so step 0 measures the heading from the local
    anchor [0, 0] to the first waypoint.

    :param waypoints: [T, 2] trajectory in local frame.
    :return: [T] array of headings in radians.
    """
    origin = np.zeros((1, 2))
    pts = np.vstack([origin, waypoints])
    deltas = np.diff(pts, axis=0)
    return np.arctan2(deltas[:, 1], deltas[:, 0])


def ego_ahe(pred_waypoints: np.ndarray, future_gt: np.ndarray) -> float:
    """
    Average Heading Error (rad) — mean absolute angular difference between
    predicted and ground-truth heading at every timestep.

    Heading at step t is computed from the displacement pred[t] - pred[t-1],
    with pred[0] measured from the local origin [0, 0].

    :param pred_waypoints: [T, 2] predicted trajectory in local frame.
    :param future_gt:      [T, 2] ground-truth future in local frame.
    :return: Mean absolute heading error in radians.
    """
    pred_h = _headings_from_waypoints(pred_waypoints)
    gt_h = _headings_from_waypoints(future_gt)
    diff = (pred_h - gt_h + np.pi) % (2 * np.pi) - np.pi   # wrap to [-pi, pi]
    return float(np.mean(np.abs(diff)))


def ego_fhe(pred_waypoints: np.ndarray, future_gt: np.ndarray) -> float:
    """
    Final Heading Error (rad) — absolute angular difference between predicted
    and ground-truth heading at the last timestep only.

    :param pred_waypoints: [T, 2] predicted trajectory in local frame.
    :param future_gt:      [T, 2] ground-truth future in local frame.
    :return: Absolute heading error at the final step in radians.
    """
    pred_h = _headings_from_waypoints(pred_waypoints)
    gt_h = _headings_from_waypoints(future_gt)
    diff = (pred_h[-1] - gt_h[-1] + np.pi) % (2 * np.pi) - np.pi
    return float(abs(diff))


def ego_speed_error(pred_waypoints: np.ndarray, future_gt: np.ndarray) -> float:
    """
    Mean Absolute Speed Error (m / step) between predicted and ground-truth
    trajectories.

    Speed at each step is derived from the displacement between consecutive
    waypoints (including the step from the origin [0, 0] to the first
    waypoint). This measures how well the model captures longitudinal
    dynamics, independent of heading error.

    :param pred_waypoints: [T, 2] predicted trajectory in local frame.
    :param future_gt:      [T, 2] ground-truth future in local frame.
    :return: Mean absolute speed error in metres per timestep (0.5 s).
    """
    origin = np.zeros((1, 2))
    pred_with_origin = np.vstack([origin, pred_waypoints])
    gt_with_origin = np.vstack([origin, future_gt])

    pred_speeds = np.linalg.norm(np.diff(pred_with_origin, axis=0), axis=1)
    gt_speeds = np.linalg.norm(np.diff(gt_with_origin, axis=0), axis=1)

    return float(np.mean(np.abs(pred_speeds - gt_speeds)))


def ego_offroad(pred_traj_world: np.ndarray, nusc_map: NuScenesMap) -> float:
    """
    Fraction of predicted waypoints (excluding the origin) that fall outside
    the drivable area.

    :param pred_traj_world: [T, 2] predicted trajectory in world (x, y) coordinates.
    :param nusc_map:        NuScenesMap for the current scene.
    :return: Off-road fraction in [0, 1].
    """
    if len(pred_traj_world) < 2:
        return 0.0
    offroad = sum(
        1 for j in range(1, len(pred_traj_world))
        if not _is_in_drivable_area(nusc_map, pred_traj_world[j, 0], pred_traj_world[j, 1])
    )
    return float(offroad) / (len(pred_traj_world) - 1)


def ego_offroad_rate(pred_traj_world: np.ndarray, nusc_map: NuScenesMap) -> float:
    """
    Binary off-road indicator: 1.0 if **any** predicted waypoint (excluding the
    origin) falls outside the drivable area, otherwise 0.0.

    Complements :func:`ego_offroad`, which reports the fraction of off-road
    waypoints. Use this when a single excursion from the drivable surface
    should count as a full failure (e.g. safety-style scoring).

    :param pred_traj_world: [T, 2] predicted trajectory in world (x, y) coordinates.
    :param nusc_map:        NuScenesMap for the current scene.
    :return: 1.0 if at least one waypoint is off-road, else 0.0.
    """
    if len(pred_traj_world) < 2:
        return 0.0
    for j in range(1, len(pred_traj_world)):
        if not _is_in_drivable_area(nusc_map, pred_traj_world[j, 0], pred_traj_world[j, 1]):
            return 1.0
    return 0.0


def ego_offyaw(pred_traj_world: np.ndarray, nusc_map: NuScenesMap,
               threshold_rad: float = _OFF_YAW_THRESHOLD_RAD) -> float:
    """
    Mean angular deviation (rad) between predicted heading and lane heading.

    Intersection segments are excluded (the yaw constraint does not apply
    there). Deviations below ``threshold_rad`` (default 45 deg) are counted
    as zero, so only meaningfully misaligned segments contribute.

    :param pred_traj_world: [T, 2] predicted trajectory in world (x, y) coordinates.
    :param nusc_map:        NuScenesMap for the current scene.
    :param threshold_rad:   Deviations strictly below this are treated as zero.
    :return: Mean off-yaw value in radians.
    """
    if len(pred_traj_world) < 2:
        return 0.0
    contributions = []
    for j in range(len(pred_traj_world) - 1):
        mx = 0.5 * (pred_traj_world[j, 0] + pred_traj_world[j + 1, 0])
        my = 0.5 * (pred_traj_world[j, 1] + pred_traj_world[j + 1, 1])
        if _is_in_intersection(nusc_map, mx, my):
            contributions.append(0.0)
            continue
        dx = pred_traj_world[j + 1, 0] - pred_traj_world[j, 0]
        dy = pred_traj_world[j + 1, 1] - pred_traj_world[j, 1]
        if dx == 0.0 and dy == 0.0:
            contributions.append(0.0)
            continue
        theta_traj = np.arctan2(dy, dx)
        lane_orient = _projected_lane_heading(nusc_map, mx, my, theta_query=theta_traj)
        if lane_orient is None:
            raw_diff = np.pi / 2
        else:
            diff = (theta_traj - lane_orient + np.pi) % (2 * np.pi) - np.pi
            raw_diff = abs(diff)
        contributions.append(raw_diff if raw_diff > threshold_rad else 0.0)
    return float(np.mean(contributions))


def compute_ego_metrics(pred_waypoints: np.ndarray,
                        future_gt: np.ndarray,
                        anchor_pos: np.ndarray,
                        anchor_yaw: float,
                        nusc_map: NuScenesMap) -> Dict[str, float]:
    """
    Compute every ego-trajectory metric used by the workshop leaderboard.

    Displacement and heading metrics are computed in the local frame
    (``pred_waypoints`` and ``future_gt`` must share the same coordinate
    system, with the anchor at [0, 0] and forward = +X). Map metrics
    (``offroad``, ``offyaw``) are computed after rotating + translating the
    predicted waypoints back to world coordinates using ``anchor_pos`` /
    ``anchor_yaw``.

    Reported horizons follow the nuScenes prediction challenge convention
    at 2 Hz (0.5 s per step):
        - ade_2s      : mean L2 over the first 4 steps
        - ade_4s      : mean L2 over the first 8 steps
        - ade_6s      : mean L2 over all 12 steps (full horizon)
        - fde         : L2 at the final step (6 s)
        - miss_rate   : 1.0 if max point error >= 2 m, else 0.0
        - speed_error : mean absolute speed error (m / step)
        - ahe         : average heading error across all steps (rad)
        - fhe         : final heading error at the last step (rad)
        - offroad     : fraction of waypoints outside the drivable area
        - offroad_rate: 1.0 if any waypoint is outside the drivable area, else 0.0
        - offyaw      : mean angular deviation from lane heading (rad)

    :param pred_waypoints: [12, 2] predicted trajectory in local frame (x forward).
    :param future_gt:      [12, 2] ground-truth future in local frame.
    :param anchor_pos:     [2]     global (x, y) of the anchor frame.
    :param anchor_yaw:     float   global yaw (rad) of the anchor frame.
    :param nusc_map:       NuScenesMap for the current scene.
    :return: Dict mapping metric name to float value (see list above).
    """
    n = len(pred_waypoints)

    # --- Displacement metrics (local frame) ---
    per_step_l2 = np.linalg.norm(pred_waypoints - future_gt, axis=1)  # [T]

    steps_2s = min(4, n)
    steps_4s = min(8, n)
    steps_6s = n

    ade_2s = float(np.mean(per_step_l2[:steps_2s]))
    ade_4s = float(np.mean(per_step_l2[:steps_4s]))
    ade_6s = float(np.mean(per_step_l2[:steps_6s]))
    fde = float(per_step_l2[-1])

    miss_rate = float(np.max(per_step_l2) >= 2.0)

    speed_error = ego_speed_error(pred_waypoints, future_gt)
    ahe = ego_ahe(pred_waypoints, future_gt)
    fhe = ego_fhe(pred_waypoints, future_gt)

    # --- Map metrics (world frame) ---
    cos_y, sin_y = np.cos(anchor_yaw), np.sin(anchor_yaw)
    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    pred_world = (R @ pred_waypoints.T).T + anchor_pos

    offroad = ego_offroad(pred_world, nusc_map)
    offroad_rate = ego_offroad_rate(pred_world, nusc_map)
    offyaw = ego_offyaw(pred_world, nusc_map)

    return {
        'ade_2s':       ade_2s,
        'ade_4s':       ade_4s,
        'ade_6s':       ade_6s,
        'fde':          fde,
        'miss_rate':    miss_rate,
        'speed_error':  speed_error,
        'ahe':          ahe,
        'fhe':          fhe,
        'offroad':      offroad,
        'offroad_rate': offroad_rate,
        'offyaw':       offyaw,
    }
