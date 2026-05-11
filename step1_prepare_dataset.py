#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BƯỚC 1 — CHUẨN BỊ DATASET
==========================
Script này đọc ảnh waterfall từ thư mục captures/ (hoặc auto-capture folder),
tổ chức theo nhãn, chia train/val/test, và tạo cấu trúc thư mục chuẩn.

Cấu trúc đầu ra:
    dataset/
    ├── train/
    │   ├── uav/           ← ảnh có UAV
    │   └── no_uav/        ← ảnh không có UAV
    ├── val/
    │   ├── uav/
    │   └── no_uav/
    └── test/
        ├── uav/
        └── no_uav/

Chạy:
    python step1_prepare_dataset.py --src ../backend/captures --dst dataset --split 0.7 0.15 0.15
"""

import argparse
import random
import shutil
import sqlite3
from pathlib import Path
from collections import defaultdict


def load_from_db(db_path: Path) -> list[dict]:
    """Đọc danh sách captures từ SQLite DB."""
    if not db_path.exists():
        print(f"  [!] Không tìm thấy DB: {db_path}")
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT filepath, label_manual, ai_has_uav, ai_uav_type FROM captures"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_from_folder(folder: Path, label: str = "") -> list[dict]:
    """
    Đọc ảnh từ folder không có DB.
    Nếu folder name chứa 'uav' → label uav, ngược lại no_uav.
    """
    records = []
    for p in folder.rglob("*.png"):
        folder_lower = p.parent.name.lower()
        if label:
            lbl = label
        elif "no_uav" in folder_lower or "negative" in folder_lower:
            lbl = "no_uav"
        elif "uav" in folder_lower or "positive" in folder_lower:
            lbl = "uav"
        else:
            lbl = ""   # chưa gán nhãn
        records.append({"filepath": str(p), "label_manual": lbl, "ai_has_uav": 0})
    return records


def split_and_copy(
    records: list[dict],
    dst: Path,
    split: tuple[float, float, float],
    seed: int = 42,
):
    """Chia và copy ảnh vào train/val/test."""
    random.seed(seed)

    # Phân nhóm theo nhãn
    by_label: dict[str, list] = defaultdict(list)
    skipped = 0
    for r in records:
        label = r.get("label_manual", "").strip()
        if not label:
            # Fallback: dùng kết quả AI nếu chưa gán nhãn thủ công
            label = "uav" if r.get("ai_has_uav") else "no_uav"
        fp = Path(r["filepath"])
        if not fp.exists():
            skipped += 1
            continue
        by_label[label].append(fp)

    print(f"\n  Thống kê nhãn:")
    for lbl, files in by_label.items():
        print(f"    {lbl:20s}: {len(files):4d} ảnh")
    print(f"    Bỏ qua (file không tồn tại): {skipped}")

    tr, va, te = split
    splits = {"train": tr, "val": va, "test": te}

    total_copied = 0
    for label, files in by_label.items():
        random.shuffle(files)
        n = len(files)
        n_tr = int(n * tr)
        n_va = int(n * va)

        groups = {
            "train": files[:n_tr],
            "val":   files[n_tr : n_tr + n_va],
            "test":  files[n_tr + n_va :],
        }

        for split_name, split_files in groups.items():
            out_dir = dst / split_name / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for src in split_files:
                shutil.copy2(src, out_dir / src.name)
                total_copied += 1

    print(f"\n  Đã copy {total_copied} ảnh vào {dst}/")
    for split_name in ["train", "val", "test"]:
        for label in by_label:
            d = dst / split_name / label
            if d.exists():
                count = len(list(d.glob("*.png")))
                print(f"    {split_name}/{label}: {count} ảnh")


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dataset waterfall UAV")
    parser.add_argument("--src",   default="../backend/captures",
                        help="Thư mục nguồn (chứa DB hoặc ảnh PNG)")
    parser.add_argument("--dst",   default="dataset",
                        help="Thư mục đích")
    parser.add_argument("--split", nargs=3, type=float,
                        default=[0.70, 0.15, 0.15],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Tỉ lệ chia (mặc định: 0.70 0.15 0.15)")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--from-folder", default="",
                        help="Đọc ảnh từ folder con thay vì DB (vd: uav hoặc no_uav)")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    print(f"\n{'='*55}")
    print(f"  BƯỚC 1 — CHUẨN BỊ DATASET")
    print(f"{'='*55}")
    print(f"  Nguồn  : {src}")
    print(f"  Đích   : {dst}")
    print(f"  Split  : train={args.split[0]:.0%} val={args.split[1]:.0%} test={args.split[2]:.0%}")

    assert abs(sum(args.split) - 1.0) < 1e-6, "Tổng split phải = 1.0"

    # Ưu tiên đọc từ DB
    db_path = src / "captures.db"
    if db_path.exists():
        print(f"\n  Đọc từ DB: {db_path}")
        records = load_from_db(db_path)
        print(f"  Tìm thấy {len(records)} bản ghi")
    else:
        print(f"\n  Không có DB → đọc ảnh từ thư mục: {src}")
        records = load_from_folder(src, args.from_folder)
        print(f"  Tìm thấy {len(records)} ảnh")

    if not records:
        print("\n  [!] Không có dữ liệu. Hãy chụp ảnh waterfall trước!")
        return

    split_and_copy(records, dst, tuple(args.split), args.seed)

    print(f"\n  ✓ Dataset sẵn sàng tại: {dst}/")
    print(f"  → Tiếp theo: python step2_train.py --data {dst}\n")


if __name__ == "__main__":
    main()
