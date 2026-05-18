#!/usr/bin/env python3
"""Create OpenVINS bags with camera messages removed over a time interval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rosbag


PLATO_ROOT = Path(__file__).resolve().parents[3]


def make_dropout_bag(
    input_bag: Path,
    output_bag: Path,
    camera_topics: set[str],
    dropout_start_s: float,
    dropout_end_s: float,
    compression: str,
) -> dict[str, object]:
    output_bag.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        "kept": {},
        "dropped": {},
    }
    with rosbag.Bag(str(input_bag), "r") as src:
        bag_start = src.get_start_time()
        start_abs = bag_start + dropout_start_s
        end_abs = bag_start + dropout_end_s
        with rosbag.Bag(str(output_bag), "w", compression=compression) as dst:
            for topic, msg, t in src.read_messages():
                topic_counts = counts["dropped"] if (
                    topic in camera_topics and start_abs <= t.to_sec() < end_abs
                ) else counts["kept"]
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
                if topic_counts is counts["dropped"]:
                    continue
                dst.write(topic, msg, t)
    return {
        "input_bag": str(input_bag),
        "output_bag": str(output_bag),
        "dropout_start_s": dropout_start_s,
        "dropout_end_s": dropout_end_s,
        "counts": counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=PLATO_ROOT
        / "results"
        / "openvins_eval"
        / "LIASuw"
        / "phys_delta_vo_framefixed_clean_imu"
        / "0205s14_80",
    )
    parser.add_argument("--methods", nargs="+", default=["constant", "network"])
    parser.add_argument("--dropout-start-s", type=float, default=5.0)
    parser.add_argument("--dropout-end-s", type=float, default=75.0)
    parser.add_argument("--compression", default="bz2")
    parser.add_argument("--cam0-topic", default="/cam0/image_raw")
    parser.add_argument("--cam1-topic", default="/cam1/image_raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tag = f"dropout_{args.dropout_start_s:g}_{args.dropout_end_s:g}"
    output_dir = args.sequence_dir / f"bags_{tag}"
    summaries = []
    for method in args.methods:
        summaries.append(
            make_dropout_bag(
                args.sequence_dir / "bags" / f"{method}.bag",
                output_dir / f"{method}.bag",
                {args.cam0_topic, args.cam1_topic},
                args.dropout_start_s,
                args.dropout_end_s,
                args.compression,
            )
        )
    summary_path = output_dir / "bag_validation.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(summary_path)
    for item in summaries:
        print(
            f"{Path(item['output_bag']).name}: "
            f"dropped={item['counts']['dropped']} kept={item['counts']['kept']}"
        )


if __name__ == "__main__":
    main()
