#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VeReMi NextGen highway_2 数据扁平化处理脚本

功能：
1. 读取若干“外层 ZIP -> Train/Validation/Test -> 内层 ZIP -> 车辆 JSON”的嵌套数据；
2. 将每条接收到的 CAM 消息展开为机器学习表格中的一行；
3. 拆分位置向量、统一数据类型、保留官方数据划分；
4. 构造二分类标签 is_attack 和多分类标签 attack_type；
5. 输出完整 Parquet 主表、分层 CSV 预览和数据处理日志。

运行示例：
python highway2_数据处理脚本.py --input-dir /path/to/zips --output-dir /path/to/output

说明：
- baseline 数据没有 attacker 字段，本脚本将 is_attack 设为 0，并用 attacker_source 记录该标签来自 baseline 规则；
- 攻击子集中 attacker=0 的消息在多分类标签中设为 benign；
- source_attack_subset 始终保留数据包来源，不能等同于消息级 attack_type；
- 本脚本不删除攻击造成的异常值，也不进行缺失值填补、标准化或类别平衡。
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


EXPECTED_ATTACKS = [
    "constantPositionOffset",
    "randomPositionOffset",
    "positionMirroring",
    "constantSpeedOffset",
    "randomSpeedOffset",
    "zeroSpeedReport",
    "suddenConstantSpeed",
    "reversedHeading",
    "feignedBraking",
    "accelerationMultiplication",
    "timeDelayAttack",
    "dosAttack",
    "trafficCongestionSybil",
    "suddenStop",
    "dataReplay",
]

SPLITS = {"train": "train", "validation": "validation", "test": "test"}

COLUMNS = [
    # 数据来源与追踪信息
    "dataset_name",
    "scenario",
    "environment",
    "traffic_period",
    "traffic_density",
    "split",
    "source_attack_subset",
    "source_outer_zip",
    "source_inner_zip",
    "source_file",
    "receiver_id",
    # 消息原始字段
    "rcvTime",
    "sendTime",
    "delay_ns",
    "sender_id",
    "sender_alias",
    "messageID",
    "attacker_raw",
    "attacker_source",
    # 接收车辆状态
    "receiver_pos_x",
    "receiver_pos_y",
    "receiver_pos_z",
    "receiver_pos_noise_x",
    "receiver_pos_noise_y",
    "receiver_pos_noise_z",
    "receiver_spd",
    "receiver_spd_noise",
    "receiver_acl",
    "receiver_acl_noise",
    "receiver_hed",
    "receiver_hed_noise",
    "receiver_driversProfile",
    # 发送车辆状态
    "sender_pos_x",
    "sender_pos_y",
    "sender_pos_z",
    "sender_pos_noise_x",
    "sender_pos_noise_y",
    "sender_pos_noise_z",
    "sender_spd",
    "sender_spd_noise",
    "sender_acl",
    "sender_acl_noise",
    "sender_hed",
    "sender_hed_noise",
    "sender_driversProfile",
    "sender_distance_to_road_edge",
    # 派生字段与标签
    "sender_receiver_distance_m",
    "is_attack",
    "attack_type",
]

NUMERIC_COLUMNS = [
    "rcvTime",
    "sendTime",
    "delay_ns",
    "sender_alias",
    "messageID",
    "attacker_raw",
    "receiver_pos_x",
    "receiver_pos_y",
    "receiver_pos_z",
    "receiver_pos_noise_x",
    "receiver_pos_noise_y",
    "receiver_pos_noise_z",
    "receiver_spd",
    "receiver_spd_noise",
    "receiver_acl",
    "receiver_acl_noise",
    "receiver_hed",
    "receiver_hed_noise",
    "sender_pos_x",
    "sender_pos_y",
    "sender_pos_z",
    "sender_pos_noise_x",
    "sender_pos_noise_y",
    "sender_pos_noise_z",
    "sender_spd",
    "sender_spd_noise",
    "sender_acl",
    "sender_acl_noise",
    "sender_hed",
    "sender_hed_noise",
    "sender_distance_to_road_edge",
    "sender_receiver_distance_m",
    "is_attack",
]


def normalize_zip_path(path: str) -> str:
    """统一 ZIP 内可能存在的 Windows 反斜杠路径。"""
    return path.replace("\\", "/")


def infer_subset_name(zip_path: Path) -> str:
    """从外层压缩包名称提取攻击子集名。"""
    stem = zip_path.name.strip()
    if stem.lower().endswith(".zip"):
        stem = stem[:-4].strip()
    if stem == "InTAS_highway_2":
        return "baseline"
    prefix = "InTAS_highway_2_"
    if stem.startswith(prefix):
        return stem[len(prefix):].strip()
    return stem


def infer_split(inner_zip_name: str) -> str:
    """从内层压缩包路径中提取 split。"""
    parts = normalize_zip_path(inner_zip_name).split("/")
    for part in parts:
        value = part.lower()
        if value in SPLITS:
            return SPLITS[value]
    return "unknown"


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_xyz(value: Any, prefix: str, row: dict[str, Any], counter: Counter) -> None:
    """把 'x,y,z' 字符串或长度为 3 的列表拆成三个数值列。"""
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = []

    if len(parts) < 3:
        counter[f"bad_{prefix}"] += 1
        xyz = [None, None, None]
    else:
        xyz = [to_float(parts[0]), to_float(parts[1]), to_float(parts[2])]
        if any(v is None for v in xyz):
            counter[f"bad_{prefix}"] += 1

    row[f"{prefix}_x"] = xyz[0]
    row[f"{prefix}_y"] = xyz[1]
    row[f"{prefix}_z"] = xyz[2]


def flatten_message(
    msg: dict[str, Any],
    *,
    subset: str,
    split: str,
    outer_zip_name: str,
    inner_zip_name: str,
    source_file: str,
    receiver_id: str,
    counter: Counter,
) -> dict[str, Any]:
    """将一个 JSON 消息对象展开为一行。"""
    receiver = msg.get("receiver") or {}
    sender = msg.get("sender") or {}

    row: dict[str, Any] = {
        "dataset_name": "VeReMi NextGen",
        "scenario": "highway_2",
        "environment": "highway",
        "traffic_period": "2 AM",
        "traffic_density": "low",
        "split": split,
        "source_attack_subset": subset,
        "source_outer_zip": outer_zip_name,
        "source_inner_zip": normalize_zip_path(inner_zip_name),
        "source_file": normalize_zip_path(source_file),
        "receiver_id": receiver_id,
        "rcvTime": to_int(msg.get("rcvTime")),
        "sendTime": to_int(msg.get("sendTime")),
        "sender_id": msg.get("sender_id"),
        "sender_alias": to_int(msg.get("sender_alias")),
        "messageID": to_int(msg.get("messageID")),
        "attacker_raw": to_int(msg.get("attacker")),
        "receiver_spd": to_float(receiver.get("spd")),
        "receiver_spd_noise": to_float(receiver.get("spd_noise")),
        "receiver_acl": to_float(receiver.get("acl")),
        "receiver_acl_noise": to_float(receiver.get("acl_noise")),
        "receiver_hed": to_float(receiver.get("hed")),
        "receiver_hed_noise": to_float(receiver.get("hed_noise")),
        "receiver_driversProfile": receiver.get("driversProfile"),
        "sender_spd": to_float(sender.get("spd")),
        "sender_spd_noise": to_float(sender.get("spd_noise")),
        "sender_acl": to_float(sender.get("acl")),
        "sender_acl_noise": to_float(sender.get("acl_noise")),
        "sender_hed": to_float(sender.get("hed")),
        "sender_hed_noise": to_float(sender.get("hed_noise")),
        "sender_driversProfile": sender.get("driversProfile"),
        "sender_distance_to_road_edge": to_float(sender.get("distance_to_road_edge")),
    }

    if row["rcvTime"] is not None and row["sendTime"] is not None:
        row["delay_ns"] = row["rcvTime"] - row["sendTime"]
    else:
        row["delay_ns"] = None
        counter["missing_time"] += 1

    parse_xyz(receiver.get("pos"), "receiver_pos", row, counter)
    parse_xyz(receiver.get("pos_noise"), "receiver_pos_noise", row, counter)
    parse_xyz(sender.get("pos"), "sender_pos", row, counter)
    parse_xyz(sender.get("pos_noise"), "sender_pos_noise", row, counter)

    # 标签规则
    if subset == "baseline":
        row["attacker_source"] = "baseline_assigned_zero"
        row["is_attack"] = 0
        row["attack_type"] = "benign"
        counter["baseline_label_assigned"] += 1
    elif row["attacker_raw"] is None:
        row["attacker_source"] = "missing_in_message"
        row["is_attack"] = None
        row["attack_type"] = None
        counter["missing_attacker"] += 1
    else:
        row["attacker_source"] = "message_field"
        row["is_attack"] = row["attacker_raw"]
        row["attack_type"] = subset if row["attacker_raw"] == 1 else "benign"

    # 发送车与接收车的欧氏距离
    values = [
        row.get("sender_pos_x"),
        row.get("sender_pos_y"),
        row.get("sender_pos_z"),
        row.get("receiver_pos_x"),
        row.get("receiver_pos_y"),
        row.get("receiver_pos_z"),
    ]
    if any(v is None for v in values):
        row["sender_receiver_distance_m"] = None
        counter["missing_distance_inputs"] += 1
    else:
        row["sender_receiver_distance_m"] = float(
            np.sqrt(
                (values[0] - values[3]) ** 2
                + (values[1] - values[4]) ** 2
                + (values[2] - values[5]) ** 2
            )
        )

    return {column: row.get(column) for column in COLUMNS}


def process_outer_zip(
    outer_zip_path: Path,
    parts_dir: Path,
    log_rows: list[dict[str, Any]],
) -> None:
    """处理一个外层攻击数据包。"""
    subset = infer_subset_name(outer_zip_path)

    with zipfile.ZipFile(outer_zip_path) as outer_zip:
        for inner_zip_name in outer_zip.namelist():
            normalized_inner_name = normalize_zip_path(inner_zip_name)
            if not normalized_inner_name.lower().endswith(".zip"):
                continue
            if "__MACOSX" in normalized_inner_name:
                continue

            split = infer_split(normalized_inner_name)
            inner_bytes = outer_zip.read(inner_zip_name)
            counter: Counter = Counter()
            rows: list[dict[str, Any]] = []
            json_files = 0
            json_read_errors = 0

            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zip:
                for source_file in inner_zip.namelist():
                    normalized_file = normalize_zip_path(source_file)
                    if not normalized_file.lower().endswith(".json"):
                        continue
                    if "__MACOSX" in normalized_file:
                        continue

                    json_files += 1
                    receiver_id = Path(normalized_file).stem
                    try:
                        messages = json.loads(inner_zip.read(source_file))
                        if isinstance(messages, dict):
                            messages = [messages]
                    except Exception:
                        json_read_errors += 1
                        continue

                    for msg in messages:
                        if not isinstance(msg, dict):
                            counter["non_dict_message"] += 1
                            continue
                        rows.append(
                            flatten_message(
                                msg,
                                subset=subset,
                                split=split,
                                outer_zip_name=outer_zip_path.name,
                                inner_zip_name=normalized_inner_name,
                                source_file=normalized_file,
                                receiver_id=receiver_id,
                                counter=counter,
                            )
                        )

            df = pd.DataFrame(rows, columns=COLUMNS)
            for column in NUMERIC_COLUMNS:
                df[column] = pd.to_numeric(df[column], errors="coerce")

            part_path = parts_dir / f"{subset}__{split}.parquet"
            df.to_parquet(part_path, index=False, compression="snappy")

            label_counts = df["is_attack"].value_counts(dropna=False).to_dict()
            attack_counts = df["attack_type"].value_counts(dropna=False).to_dict()
            log_rows.append(
                {
                    "subset": subset,
                    "split": split,
                    "outer_zip": outer_zip_path.name,
                    "inner_zip": normalized_inner_name,
                    "json_files": json_files,
                    "json_read_errors": json_read_errors,
                    "messages_written": int(len(df)),
                    "is_attack_0": int(label_counts.get(0, 0)),
                    "is_attack_1": int(label_counts.get(1, 0)),
                    "is_attack_missing": int(df["is_attack"].isna().sum()),
                    "attack_type_counts": json.dumps(attack_counts, ensure_ascii=False, sort_keys=True),
                    "baseline_label_assigned": int(counter.get("baseline_label_assigned", 0)),
                    "bad_receiver_pos": int(counter.get("bad_receiver_pos", 0)),
                    "bad_receiver_pos_noise": int(counter.get("bad_receiver_pos_noise", 0)),
                    "bad_sender_pos": int(counter.get("bad_sender_pos", 0)),
                    "bad_sender_pos_noise": int(counter.get("bad_sender_pos_noise", 0)),
                    "missing_time": int(counter.get("missing_time", 0)),
                    "missing_distance_inputs": int(counter.get("missing_distance_inputs", 0)),
                    "part_file": str(part_path),
                }
            )
            print(
                f"完成 {subset:28s} {split:10s} "
                f"文件={json_files:4d} 行={len(df):7d} "
                f"攻击={int(label_counts.get(1, 0)):7d} "
                f"非攻击={int(label_counts.get(0, 0)):7d}",
                flush=True,
            )


def combine_parts(parts_dir: Path, output_path: Path) -> None:
    """把分区 Parquet 合并成一个主表；先写临时盘，再复制到输出目录。"""
    part_files = sorted(p for p in parts_dir.glob("*.parquet") if not p.name.startswith("_combined"))
    if not part_files:
        raise RuntimeError("没有可合并的 Parquet 分区。")

    dataset = ds.dataset([str(p) for p in part_files], format="parquet")
    temp_output = parts_dir / "_combined.parquet"
    if temp_output.exists():
        temp_output.unlink()
    with pq.ParquetWriter(temp_output, dataset.schema, compression="snappy") as writer:
        for batch in dataset.to_batches(batch_size=100_000):
            writer.write_batch(batch)

    if output_path.exists():
        output_path.unlink()
    shutil.copyfile(temp_output, output_path)


def make_preview(parts_dir: Path, preview_path: Path, per_part: int = 250, seed: int = 42) -> None:
    """从每个 subset-split 分区抽样，生成便于检查的 CSV 预览。"""
    samples = []
    for part in sorted(parts_dir.glob("*.parquet")):
        df = pd.read_parquet(part)
        n = min(per_part, len(df))
        if n > 0:
            samples.append(df.sample(n=n, random_state=seed))
    preview = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    preview.to_csv(preview_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="处理 VeReMi NextGen highway_2 嵌套 JSON 数据。")
    parser.add_argument("--input-dir", required=True, help="包含 highway_2 外层 ZIP 的目录。")
    parser.add_argument("--output-dir", required=True, help="结果输出目录。")
    parser.add_argument(
        "--additional-zip",
        action="append",
        default=[],
        help="额外补充的 ZIP 路径；可重复传入。",
    )
    parser.add_argument("--preview-per-part", type=int, default=250, help="每个分区抽入 CSV 预览的行数。")
    parser.add_argument("--work-dir", default=None, help="中间文件目录；默认使用系统临时目录。")
    args = parser.parse_args()

    started = time.time()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if args.work_dir:
        work_root = Path(args.work_dir)
        work_root.mkdir(parents=True, exist_ok=True)
    else:
        work_root = Path(tempfile.mkdtemp(prefix="veremi_highway2_"))
    work_dir = work_root / "parts"
    shutil.rmtree(work_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    zip_paths = sorted(input_dir.glob("InTAS_highway_2*.zip"))
    zip_paths.extend(Path(p) for p in args.additional_zip)
    zip_paths = [p for p in zip_paths if p.exists() and zipfile.is_zipfile(p)]
    zip_paths = sorted(set(zip_paths), key=lambda p: infer_subset_name(p))

    found_subsets = {infer_subset_name(p) for p in zip_paths}
    expected_subsets = {"baseline", *EXPECTED_ATTACKS}
    missing_subsets = sorted(expected_subsets - found_subsets)
    extra_subsets = sorted(found_subsets - expected_subsets)

    print(f"发现外层数据包：{len(zip_paths)} 个")
    print(f"发现数据子集：{len(found_subsets)} 个")
    if missing_subsets:
        print("警告：缺少子集：" + ", ".join(missing_subsets))
    if extra_subsets:
        print("提示：发现额外子集：" + ", ".join(extra_subsets))

    log_rows: list[dict[str, Any]] = []
    for zip_path in zip_paths:
        process_outer_zip(zip_path, work_dir, log_rows)

    parquet_path = output_dir / "highway2_机器学习主表.parquet"
    preview_path = output_dir / "highway2_数据预览.csv"
    log_path = output_dir / "highway2_处理日志.csv"
    summary_path = output_dir / "highway2_处理摘要.json"

    combine_parts(work_dir, parquet_path)
    make_preview(work_dir, preview_path, per_part=args.preview_per_part)

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "outer_zip_count": len(zip_paths),
        "subset_count": len(found_subsets),
        "expected_subset_count": len(expected_subsets),
        "missing_subsets": missing_subsets,
        "extra_subsets": extra_subsets,
        "part_count": len(log_rows),
        "total_rows": int(log_df["messages_written"].sum()) if not log_df.empty else 0,
        "total_json_files": int(log_df["json_files"].sum()) if not log_df.empty else 0,
        "total_json_read_errors": int(log_df["json_read_errors"].sum()) if not log_df.empty else 0,
        "total_is_attack_0": int(log_df["is_attack_0"].sum()) if not log_df.empty else 0,
        "total_is_attack_1": int(log_df["is_attack_1"].sum()) if not log_df.empty else 0,
        "elapsed_seconds": round(time.time() - started, 3),
        "outputs": {
            "main_parquet": str(parquet_path),
            "preview_csv": str(preview_path),
            "processing_log": str(log_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n处理完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
