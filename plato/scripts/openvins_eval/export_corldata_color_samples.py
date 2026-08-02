#!/usr/bin/env python3
"""Export representative corldata color frames from a ROS bag."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge


PLATO_ROOT = Path(__file__).resolve().parents[3]


def export_samples(args: argparse.Namespace) -> Path:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("light_on_01", 5.0, "light on"),
        ("light_on_02", 25.0, "light on"),
        ("light_off_01", 18.0, "light off"),
        ("light_off_02", 35.0, "light off"),
    ]

    bridge = CvBridge()
    frames = []
    with rosbag.Bag(str(args.bag)) as bag:
        first_t = None
        for _, msg, t in bag.read_messages(topics=[args.topic]):
            stamp = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() else t.to_sec()
            if first_t is None:
                first_t = t.to_sec()
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frames.append(
                {
                    "bag_time_s": t.to_sec(),
                    "header_time_s": stamp,
                    "rel_s": t.to_sec() - first_t,
                    "mean_bgr": float(img.mean()),
                    "encoding": msg.encoding,
                    "image": img,
                }
            )
    if not frames:
        raise ValueError(f"No frames found on {args.topic} in {args.bag}")

    records = []
    selected = []
    for name, target_rel_s, state in targets:
        idx = min(range(len(frames)), key=lambda i: abs(frames[i]["rel_s"] - target_rel_s))
        item = frames[idx]
        img = item["image"]
        image_path = out_dir / f"{args.sequence}_{name}.png"
        cv2.imwrite(str(image_path), img)
        record = {
            "name": name,
            "state": state,
            "target_rel_s": target_rel_s,
            "selected_rel_s": item["rel_s"],
            "bag_time_s": item["bag_time_s"],
            "header_time_s": item["header_time_s"],
            "mean_bgr": item["mean_bgr"],
            "encoding": item["encoding"],
            "path": str(image_path),
        }
        records.append(record)
        selected.append((record, img))

    thumbs = []
    for record, img in selected:
        h, w = img.shape[:2]
        scale = args.thumb_width / w
        resized = cv2.resize(img, (args.thumb_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        canvas = cv2.copyMakeBorder(resized, 52, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        label = f"{record['state']}  t={record['selected_rel_s']:.1f}s  mean={record['mean_bgr']:.1f}"
        cv2.putText(canvas, label, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
        thumbs.append(canvas)

    montage = np.vstack([np.hstack([thumbs[0], thumbs[1]]), np.hstack([thumbs[2], thumbs[3]])])
    montage_path = out_dir / f"{args.sequence}_color_light_on_off_samples.png"
    cv2.imwrite(str(montage_path), montage)

    csv_path = out_dir / f"{args.sequence}_color_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    json_path = out_dir / f"{args.sequence}_color_samples.json"
    json_path.write_text(
        json.dumps({"bag": str(args.bag), "topic": args.topic, "montage": str(montage_path), "samples": records}, indent=2),
        encoding="utf-8",
    )
    return montage_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="S15")
    parser.add_argument("--bag", type=Path, default=Path("/home/amao/datasets/corldata/0511_data/S15.bag"))
    parser.add_argument("--topic", default="/camera/color/image_raw")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLATO_ROOT / "results" / "pseudo_gt" / "corldata" / "S15" / "color_samples",
    )
    parser.add_argument("--thumb-width", type=int, default=420)
    args = parser.parse_args()
    montage_path = export_samples(args)
    print(montage_path)


if __name__ == "__main__":
    main()
