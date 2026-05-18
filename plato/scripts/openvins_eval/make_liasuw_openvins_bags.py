#!/usr/bin/env python3
"""Build ROS bags for OpenVINS from LIASuw images and exported IMU CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import rosbag
from genpy import Time
from sensor_msgs.msg import Image, Imu


PLATO_ROOT = Path(__file__).resolve().parents[3]


def ros_time_from_ns(timestamp_ns: int) -> Time:
    timestamp_ns = int(timestamp_ns)
    return Time(timestamp_ns // 1_000_000_000, timestamp_ns % 1_000_000_000)


def read_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[int] = []
    values: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            timestamps.append(int(row[0].strip()))
            values.append([float(x) for x in row[1:7]])
    if not timestamps:
        raise ValueError(f"No IMU rows found in {path}")
    return np.asarray(timestamps, dtype=np.int64), np.asarray(values, dtype=np.float64)


def read_image_timestamps(path: Path) -> list[int]:
    timestamps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            timestamps.append(int(stripped))
    if not timestamps:
        raise ValueError(f"No image timestamps found in {path}")
    return timestamps


def read_image_drop_intervals(path: Path | None, sequence: str, states: set[str]) -> list[tuple[float, float, str]]:
    if path is None or not states:
        return []
    intervals: list[tuple[float, float, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sequence"] != sequence or row["state"] not in states:
                continue
            intervals.append((float(row["start_image_time_s"]), float(row["end_image_time_s"]), row["state"]))
    return intervals


def drop_state_for_image_time(timestamp_ns: int, intervals: list[tuple[float, float, str]]) -> str | None:
    timestamp_s = float(timestamp_ns) * 1.0e-9
    for start_s, end_s, state in intervals:
        if start_s <= timestamp_s <= end_s:
            return state
    return None


def make_image_msg(path: Path, timestamp_ns: int, frame_id: str) -> Image:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    if image.dtype != np.uint8:
        raise ValueError(f"{path} is not uint8 grayscale")
    msg = Image()
    msg.header.stamp = ros_time_from_ns(timestamp_ns)
    msg.header.frame_id = frame_id
    msg.height = int(image.shape[0])
    msg.width = int(image.shape[1])
    msg.encoding = "mono8"
    msg.is_bigendian = 0
    msg.step = int(image.shape[1])
    msg.data = image.tobytes()
    return msg


def make_imu_msg(timestamp_ns: int, values: np.ndarray, imu_topic_frame: str) -> Imu:
    msg = Imu()
    msg.header.stamp = ros_time_from_ns(timestamp_ns)
    msg.header.frame_id = imu_topic_frame
    msg.angular_velocity.x = float(values[0])
    msg.angular_velocity.y = float(values[1])
    msg.angular_velocity.z = float(values[2])
    msg.linear_acceleration.x = float(values[3])
    msg.linear_acceleration.y = float(values[4])
    msg.linear_acceleration.z = float(values[5])
    return msg


def image_events(
    sequence_dir: Path,
    max_images: int,
    imu_start_ns: int,
    imu_end_ns: int,
    camera_imu_timeshift_s: float,
    crop_to_imu: bool,
    drop_intervals: list[tuple[float, float, str]],
) -> tuple[list[tuple[int, str, Path]], dict[str, int]]:
    left_dir = sequence_dir / "left"
    right_dir = sequence_dir / "right"
    timestamps = read_image_timestamps(sequence_dir / "timestamps.txt")
    if max_images > 0:
        timestamps = timestamps[:max_images]
    events: list[tuple[int, str, Path]] = []
    shift_ns = int(round(camera_imu_timeshift_s * 1e9))
    skipped_before = 0
    skipped_after = 0
    dropped_by_state: dict[str, int] = {}
    for ts in timestamps:
        imu_time_for_image = ts + shift_ns
        if crop_to_imu and imu_time_for_image < imu_start_ns:
            skipped_before += 1
            continue
        if crop_to_imu and imu_time_for_image > imu_end_ns:
            skipped_after += 1
            continue
        drop_state = drop_state_for_image_time(ts, drop_intervals)
        if drop_state is not None:
            dropped_by_state[drop_state] = dropped_by_state.get(drop_state, 0) + 1
            continue
        left = left_dir / f"{ts}.png"
        right = right_dir / f"{ts}.png"
        if not left.exists():
            raise FileNotFoundError(left)
        if not right.exists():
            raise FileNotFoundError(right)
        events.append((ts, "cam0", left))
        events.append((ts, "cam1", right))
    stats = {
        "input_stereo_pairs": len(timestamps),
        "kept_stereo_pairs": len(events) // 2,
        "skipped_before_imu": skipped_before,
        "skipped_after_imu": skipped_after,
        "dropped_stereo_pairs": int(sum(dropped_by_state.values())),
        "camera_imu_timeshift_ns": shift_ns,
    }
    for state, count in sorted(dropped_by_state.items()):
        stats[f"dropped_{state}_stereo_pairs"] = int(count)
    return events, stats


def write_method_bag(
    prepared_sequence_dir: Path,
    source_sequence_dir: Path,
    method: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    imu_csv = prepared_sequence_dir / "methods" / method / "mav0" / "imu0" / "data.csv"
    if not imu_csv.exists():
        raise FileNotFoundError(imu_csv)
    imu_t_ns, imu_values = read_imu_csv(imu_csv)
    if args.max_imu > 0:
        imu_t_ns = imu_t_ns[: args.max_imu]
        imu_values = imu_values[: args.max_imu]

    events: list[tuple[int, str, int | Path]] = []
    for idx, timestamp_ns in enumerate(imu_t_ns):
        events.append((int(timestamp_ns), "imu", idx))
    sequence_name = args.sequence or source_sequence_dir.name
    drop_intervals = read_image_drop_intervals(args.light_segments, sequence_name, set(args.drop_image_states))
    image_event_list, image_stats = image_events(
        source_sequence_dir,
        args.max_images,
        int(imu_t_ns[0]),
        int(imu_t_ns[-1]),
        args.camera_imu_timeshift_s,
        not args.no_crop_images_to_imu,
        drop_intervals,
    )
    events.extend(image_event_list)
    events.sort(key=lambda item: (item[0], item[1]))

    output_dir.mkdir(parents=True, exist_ok=True)
    bag_path = output_dir / f"{method}.bag"
    counts = {"imu": 0, "cam0": 0, "cam1": 0}
    with rosbag.Bag(str(bag_path), "w", compression=args.compression) as bag:
        for timestamp_ns, kind, payload in events:
            stamp = ros_time_from_ns(timestamp_ns)
            if kind == "imu":
                msg = make_imu_msg(timestamp_ns, imu_values[int(payload)], args.imu_frame)
                bag.write(args.imu_topic, msg, t=stamp)
                counts["imu"] += 1
            elif kind == "cam0":
                msg = make_image_msg(Path(payload), timestamp_ns, args.cam0_frame)
                bag.write(args.cam0_topic, msg, t=stamp)
                counts["cam0"] += 1
            elif kind == "cam1":
                msg = make_image_msg(Path(payload), timestamp_ns, args.cam1_frame)
                bag.write(args.cam1_topic, msg, t=stamp)
                counts["cam1"] += 1
            else:
                raise ValueError(f"Unknown event kind: {kind}")

    info = {
        "method": method,
        "bag": str(bag_path),
        "imu_csv": str(imu_csv),
        "counts": counts,
        "start_ns": int(events[0][0]),
        "end_ns": int(events[-1][0]),
        "duration_s": float((events[-1][0] - events[0][0]) * 1e-9),
        "image_selection": image_stats,
        "image_drop_sequence": sequence_name,
        "image_drop_states": list(args.drop_image_states),
        "image_drop_intervals": [
            {"start_image_time_s": start, "end_image_time_s": end, "state": state}
            for start, end, state in drop_intervals
        ],
        "topics": {
            "imu": args.imu_topic,
            "cam0": args.cam0_topic,
            "cam1": args.cam1_topic,
        },
    }
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create OpenVINS-readable ROS bags from LIASuw method IMU CSVs."
    )
    parser.add_argument(
        "--prepared-sequence-dir",
        type=Path,
        default=PLATO_ROOT
        / "results"
        / "openvins_eval"
        / "LIASuw"
        / "phys_delta_vo_framefixed_clean_imu"
        / "0205s14_80",
    )
    parser.add_argument(
        "--source-sequence-dir",
        type=Path,
        default=PLATO_ROOT / "results" / "pseudo_gt" / "LIASuw" / "0205s14_80" / "sequence",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <prepared-sequence-dir>/bags.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["raw", "constant", "network"],
    )
    parser.add_argument("--imu-topic", default="/imu0")
    parser.add_argument("--cam0-topic", default="/cam0/image_raw")
    parser.add_argument("--cam1-topic", default="/cam1/image_raw")
    parser.add_argument("--imu-frame", default="imu0")
    parser.add_argument("--cam0-frame", default="cam0")
    parser.add_argument("--cam1-frame", default="cam1")
    parser.add_argument("--compression", default="none", choices=["none", "bz2", "lz4"])
    parser.add_argument("--camera-imu-timeshift-s", type=float, default=0.00777437)
    parser.add_argument(
        "--light-segments",
        type=Path,
        default=None,
        help="Optional light segment CSV. When set with --drop-image-states, matching image pairs are omitted.",
    )
    parser.add_argument("--sequence", default="", help="Sequence name used to look up --light-segments.")
    parser.add_argument("--drop-image-states", nargs="+", default=[])
    parser.add_argument(
        "--no-crop-images-to-imu",
        action="store_true",
        help="Keep all images even if t_cam + timeshift is outside the IMU span.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Optional smoke-test image limit.")
    parser.add_argument("--max-imu", type=int, default=0, help="Optional smoke-test IMU limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.prepared_sequence_dir / "bags")
    summaries = []
    for method in args.methods:
        summary = write_method_bag(
            args.prepared_sequence_dir,
            args.source_sequence_dir,
            method,
            output_dir,
            args,
        )
        summaries.append(summary)
        counts = summary["counts"]
        print(
            f"{method}: wrote {summary['bag']} "
            f"(imu={counts['imu']}, cam0={counts['cam0']}, cam1={counts['cam1']})"
        )

    validation_path = output_dir / "bag_validation.json"
    validation_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"Bag validation: {validation_path}")


if __name__ == "__main__":
    main()
