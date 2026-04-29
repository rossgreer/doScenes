from __future__ import annotations

"""
SETUP:

1. Install:
   pip install nuscenes-devkit torch pandas pyquaternion

2. Download nuScenes and set path below

3. Place doScenes Annotations folder

4. Run:
   python doscenes_dataloader.py
"""


"""
Each dataset item = (instruction, scene)

We:
- read doScenes CSV
- map Scene Number → nuScenes scene
- load full trajectory + sensor paths
"""


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from pyquaternion import Quaternion
except Exception:
    Quaternion = None


HISTORY_LEN = 4
FUTURE_LEN = 12
MIN_SCENE_SAMPLES = HISTORY_LEN + 1 + FUTURE_LEN


@dataclass
class DoScenesRecord:
    scene_number: int
    scene_name: str
    scene_token: str
    instruction: str
    instruction_type: str
    annotator_file: str


def _clean_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _scene_name_from_number(scene_number: Union[int, str]) -> str:
    return f"scene-{int(scene_number):04d}"


def _decode_instruction_type(code: str) -> Dict[str, bool]:
    """
    doScenes CSV uses short codes like:
    - ''   : no referentiality tag / unspecified
    - 's'  : static referential
    - 'd'  : dynamic referential
    - 'sd' : both static and dynamic referential
    """
    code = (code or "").strip().lower()
    return {
        "has_static_reference": "s" in code,
        "has_dynamic_reference": "d" in code,
    }


def _yaw_from_wxyz(rotation_wxyz: Sequence[float]) -> Optional[float]:
    if Quaternion is None:
        return None
    q = Quaternion(rotation_wxyz)
    w, x, y, z = q.elements
    yaw = torch.atan2(
        torch.tensor(2.0 * (w * z + x * y)),
        torch.tensor(1.0 - 2.0 * (y * y + z * z)),
    ).item()
    return float(yaw)


class DoScenesNuScenesDataset(Dataset):
    """
    Simple row-level dataset that joins doScenes annotations to nuScenes scenes.

    One dataset item = one (instruction, scene) pair.

    Returned item keys:
      - instruction
      - instruction_type
      - has_static_reference
      - has_dynamic_reference
      - scene_number
      - scene_name
      - scene_token
      - annotator_file
      - sample_tokens
      - timestamps_us
      - ego_xy             : FloatTensor [T, 2]
      - ego_xyz            : FloatTensor [T, 3]
      - ego_yaw            : FloatTensor [T] with NaN where unavailable
      - camera_paths       : dict[channel] -> list[path]
      - lidar_paths        : list[path]

    Notes:
      - By default, images are NOT loaded. Only file paths are returned.
      - This keeps the loader simple and fast, and lets you decide your own
        image transforms later.
    """

    def __init__(
        self,
        nusc: Any,
        annotations: Union[str, Path, Sequence[Union[str, Path]]],
        camera_channels: Sequence[str] = ("CAM_FRONT",),
        include_blank_instructions: bool = False,
        allowed_scene_names: Optional[Iterable[str]] = None,
    ) -> None:
        self.nusc = nusc
        self.dataroot = Path(nusc.dataroot)
        self.camera_channels = tuple(camera_channels)
        self.include_blank_instructions = include_blank_instructions
        self.allowed_scene_names = set(allowed_scene_names) if allowed_scene_names else None

        self.scene_by_name = {scene["name"]: scene for scene in self.nusc.scene}
        self.records = self._load_annotation_records(annotations)

    def _resolve_annotation_files(
        self, annotations: Union[str, Path, Sequence[Union[str, Path]]]
    ) -> List[Path]:
        if isinstance(annotations, (str, Path)):
            path = Path(annotations)
            if path.is_dir():
                files = sorted(path.glob("*.csv"))
            else:
                files = [path]
        else:
            files = [Path(p) for p in annotations]

        if not files:
            raise FileNotFoundError("No doScenes CSV files found.")
        return files

    def _load_single_csv(self, csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        rename_map = {}
        for c in df.columns:
            c_norm = c.strip().lower()
            if c_norm == "scene number":
                rename_map[c] = "scene_number"
            elif c_norm == "instruction type":
                rename_map[c] = "instruction_type"
            elif c_norm == "instruction":
                rename_map[c] = "instruction"
        df = df.rename(columns=rename_map)

        required = {"scene_number", "instruction_type", "instruction"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

        df["scene_number"] = pd.to_numeric(df["scene_number"], errors="coerce")
        df = df.dropna(subset=["scene_number"]).copy()
        df["scene_number"] = df["scene_number"].astype(int)
        df["instruction"] = df["instruction"].map(_clean_text)
        df["instruction_type"] = df["instruction_type"].map(_clean_text)
        df["annotator_file"] = csv_path.name
        df["scene_name"] = df["scene_number"].map(_scene_name_from_number)
        return df

    def _load_annotation_records(
        self, annotations: Union[str, Path, Sequence[Union[str, Path]]]
    ) -> List[DoScenesRecord]:
        files = self._resolve_annotation_files(annotations)
        frames = [self._load_single_csv(f) for f in files]
        merged = pd.concat(frames, ignore_index=True)

        records: List[DoScenesRecord] = []
        for row in merged.itertuples(index=False):
            if not self.include_blank_instructions and not row.instruction:
                continue
            if self.allowed_scene_names and row.scene_name not in self.allowed_scene_names:
                continue
            scene = self.scene_by_name.get(row.scene_name)
            if scene is None:
                continue
            if scene.get("nbr_samples", 0) < MIN_SCENE_SAMPLES:
                continue
            records.append(
                DoScenesRecord(
                    scene_number=int(row.scene_number),
                    scene_name=row.scene_name,
                    scene_token=scene["token"],
                    instruction=row.instruction,
                    instruction_type=row.instruction_type,
                    annotator_file=row.annotator_file,
                )
            )

        if not records:
            raise ValueError(
                "No matched doScenes rows were found after joining to nuScenes scenes. "
                "Check your CSV path, split, and nuScenes version."
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _iter_scene_samples(self, scene_token: str) -> List[Dict[str, Any]]:
        scene = self.nusc.get("scene", scene_token)
        token = scene["first_sample_token"]
        samples: List[Dict[str, Any]] = []

        while token:
            sample = self.nusc.get("sample", token)
            samples.append(sample)
            token = sample["next"]

        return samples

    def _ego_pose_from_sample(self, sample: Dict[str, Any]) -> Tuple[List[float], Optional[float]]:
        lidar_sd_token = sample["data"].get("LIDAR_TOP")
        if lidar_sd_token is None:
            raise KeyError("LIDAR_TOP not found in sample['data'].")
        lidar_sd = self.nusc.get("sample_data", lidar_sd_token)
        ego_pose = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])

        xyz = ego_pose["translation"]
        yaw = _yaw_from_wxyz(ego_pose["rotation"])
        return xyz, yaw

    def _sensor_path_if_present(self, sample: Dict[str, Any], channel: str) -> Optional[str]:
        token = sample["data"].get(channel)
        if token is None:
            return None
        sd = self.nusc.get("sample_data", token)
        return str(self.dataroot / sd["filename"])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        scene_samples = self._iter_scene_samples(record.scene_token)

        timestamps_us: List[int] = []
        sample_tokens: List[str] = []
        ego_xy: List[List[float]] = []
        ego_yaw: List[float] = []
        camera_paths: Dict[str, List[str]] = {ch: [] for ch in self.camera_channels}
        lidar_paths: List[str] = []

        for sample in scene_samples:
            sample_tokens.append(sample["token"])
            timestamps_us.append(int(sample["timestamp"]))

            xyz, yaw = self._ego_pose_from_sample(sample)
            ego_xy.append([float(xyz[0]), float(xyz[1])])
            ego_yaw.append(float("nan") if yaw is None else float(yaw))

            lidar_path = self._sensor_path_if_present(sample, "LIDAR_TOP")
            lidar_paths.append(lidar_path or "")

            for ch in self.camera_channels:
                camera_paths[ch].append(self._sensor_path_if_present(sample, ch) or "")

        ego_xy_tensor = torch.tensor(ego_xy, dtype=torch.float32)
        ego_yaw_tensor = torch.tensor(ego_yaw, dtype=torch.float32)
        timestamps_tensor = torch.tensor(timestamps_us, dtype=torch.long)

        h = HISTORY_LEN
        a = HISTORY_LEN
        f_end = HISTORY_LEN + 1 + FUTURE_LEN

        out: Dict[str, Any] = {
            "instruction": record.instruction,
            "instruction_type": record.instruction_type,
            "scene_number": record.scene_number,
            "scene_name": record.scene_name,
            "scene_token": record.scene_token,
            "annotator_file": record.annotator_file,

            "history_sample_tokens": sample_tokens[:h],
            "anchor_sample_token": sample_tokens[a],
            "future_sample_tokens": sample_tokens[a + 1:f_end],

            "history_timestamps_us": timestamps_tensor[:h],
            "anchor_timestamp_us": timestamps_tensor[a],
            "future_timestamps_us": timestamps_tensor[a + 1:f_end],

            "history_xy": ego_xy_tensor[:h],
            "anchor_xy": ego_xy_tensor[a],
            "future_xy": ego_xy_tensor[a + 1:f_end],

            "history_yaw": ego_yaw_tensor[:h],
            "anchor_yaw": ego_yaw_tensor[a],
            "future_yaw": ego_yaw_tensor[a + 1:f_end],

            "history_camera_paths": {ch: paths[:h] for ch, paths in camera_paths.items()},
            "anchor_camera_paths": {ch: paths[a] for ch, paths in camera_paths.items()},
            "future_camera_paths": {ch: paths[a + 1:f_end] for ch, paths in camera_paths.items()},

            "history_lidar_paths": lidar_paths[:h],
            "anchor_lidar_path": lidar_paths[a],
            "future_lidar_paths": lidar_paths[a + 1:f_end],
        }
        out.update(_decode_instruction_type(record.instruction_type))
        return out


def doscenes_collate(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Safe default collate for variable-length scenes.
    Returns a dict of lists instead of trying to stack sequences of different lengths.
    """
    keys = batch[0].keys()
    return {k: [item[k] for item in batch] for k in keys}

def load_paths(path_file="paths.txt"):
    config = {}
    
    with open(path_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            config[key.strip()] = val.strip()
    
    # safety checks
    if "NUSCENES_ROOT" not in config:
        raise ValueError("Missing NUSCENES_ROOT in paths.txt")
    if "DOSCENES_ANNOTATIONS" not in config:
        raise ValueError("Missing DOSCENES_ANNOTATIONS in paths.txt")

    return config["NUSCENES_ROOT"], config["DOSCENES_ANNOTATIONS"]


if __name__ == "__main__":
    # pip install nuscenes-devkit torch pandas pyquaternion
    from nuscenes.nuscenes import NuScenes
    from torch.utils.data import DataLoader

    NUSCENES_ROOT, DOSCENES_ANNOTATIONS = load_paths()

    nusc = NuScenes(version="v1.0-test", dataroot=NUSCENES_ROOT, verbose=True)

    dataset = DoScenesNuScenesDataset(
        nusc=nusc,
        annotations=DOSCENES_ANNOTATIONS,
        camera_channels=("CAM_FRONT",),
        include_blank_instructions=False,
    )

    print(f"Loaded {len(dataset)} instruction-scene pairs")

    sample = dataset[0]
    print("Scene:", sample["scene_name"])
    print("Instruction:", sample["instruction"])
    print("Instruction type:", sample["instruction_type"])
    print("History xy shape:", tuple(sample["history_xy"].shape))
    print("Anchor xy:", sample["anchor_xy"])
    print("Future xy shape:", tuple(sample["future_xy"].shape))
    print("History CAM_FRONT paths:", sample["history_camera_paths"]["CAM_FRONT"])
    print("Anchor CAM_FRONT path:", sample["anchor_camera_paths"]["CAM_FRONT"])

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=doscenes_collate,
    )
    batch = next(iter(loader))
    print("Batch keys:", batch.keys())