#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chuẩn bị dataset từ cấu trúc thư mục thực tế của bạn:

  E:/bb60c/captures\
  ├── no_uav\
  │   └── noise\            → nhãn: no_uav
  └── uav\
      ├── datalink_900MHz\  → nhãn: datalink_900MHz
      ├── dji_mavic3_10M\   → nhãn: dji_mavic3_10M
      ├── dji_mavic3_20M\   → nhãn: dji_mavic3_20M
      ├── dji_mavic3_40M\   → nhãn: dji_mavic3_40M
      ├── dji_mavic3_telemetry\ → nhãn: dji_mavic3_telemetry
      ├── ELRS\             → nhãn: ELRS
      └── Futaba\           → nhãn: Futaba

Script tạo 2 dataset:

  A) BINARY (uav / no_uav) — detect có UAV không
  B) MULTICLASS (8 nhãn)   — phân loại chủng loại

Dùng:
    python prepare_your_dataset.py --src E:\\bb60c\\captures --dst E:\\bb60c\\dataset
"""

import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict


# ─── Mapping thư mục → nhãn ──────────────────────────────────────────────────
# Chỉnh sửa nếu bạn thêm loại mới
FOLDER_TO_CLASS = {
    # no_uav
    "noise":                 "no_uav",
    # uav — giữ nguyên tên thư mục làm nhãn
    "datalink_900MHz":       "datalink_900MHz",
    "dji_mavic3_10M":        "dji_mavic3_10M",
    "dji_mavic3_20M":        "dji_mavic3_20M",
    "dji_mavic3_40M":        "dji_mavic3_40M",
    "dji_mavic3_telemetry":  "dji_mavic3_telemetry",
    "ELRS":                  "ELRS",
    "Futaba":                "Futaba",
}

# Các nhãn được coi là "có UAV" (dùng cho dataset binary)
UAV_CLASSES = {
    "datalink_900MHz", "dji_mavic3_10M", "dji_mavic3_20M",
    "dji_mavic3_40M", "dji_mavic3_telemetry", "ELRS", "Futaba",
}


def scan_source(src: Path) -> list[dict]:
    """Quét toàn bộ ảnh PNG từ cấu trúc thư mục nguồn."""
    records = []
    for png in src.rglob("*.png"):
        folder = png.parent.name           # tên thư mục chứa ảnh
        label  = FOLDER_TO_CLASS.get(folder, "")
        if not label:
            # Thử tìm tên thư mục cha (phòng khi có cấu trúc lồng thêm)
            for part in reversed(png.parts):
                if part in FOLDER_TO_CLASS:
                    label = FOLDER_TO_CLASS[part]
                    break
        if label:
            records.append({"path": png, "label": label})
        else:
            print(f"  [skip] Không nhận ra nhãn: {png.relative_to(src)}")
    return records


def split_and_copy(
    records:    list[dict],
    dst:        Path,
    split:      tuple[float, float, float],
    label_key:  str,   # "label" (multiclass) hoặc "binary_label"
    seed:       int = 42,
):
    """Chia và copy ảnh vào dst/train|val|test/label/."""
    random.seed(seed)

    by_label: dict[str, list] = defaultdict(list)
    for r in records:
        by_label[r[label_key]].append(r["path"])

    print(f"\n  Phân phối ảnh:")
    for lbl in sorted(by_label):
        print(f"    {lbl:30s}: {len(by_label[lbl]):5d} ảnh")

    tr, va, te = split
    split_names = ["train", "val", "test"]
    summary: dict[str, dict] = {s: {} for s in split_names}

    for label, files in by_label.items():
        random.shuffle(files)
        n    = len(files)
        n_tr = int(n * tr)
        n_va = int(n * va)

        groups = {
            "train": files[:n_tr],
            "val":   files[n_tr: n_tr + n_va],
            "test":  files[n_tr + n_va:],
        }
        for split_name, split_files in groups.items():
            out_dir = dst / split_name / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for src_file in split_files:
                shutil.copy2(src_file, out_dir / src_file.name)
            summary[split_name][label] = len(split_files)

    # In bảng tổng kết
    all_labels = sorted(by_label.keys())
    print(f"\n  {'Nhãn':30s}  {'train':>7} {'val':>7} {'test':>7}  {'tổng':>7}")
    print(f"  {'-'*60}")
    for lbl in all_labels:
        tr_n  = summary["train"].get(lbl, 0)
        va_n  = summary["val"].get(lbl, 0)
        te_n  = summary["test"].get(lbl, 0)
        total = tr_n + va_n + te_n
        print(f"  {lbl:30s}  {tr_n:7d} {va_n:7d} {te_n:7d}  {total:7d}")
    print(f"  {'-'*60}")
    grand_total = sum(len(f) for f in by_label.values())
    print(f"  {'TỔNG':30s}  {sum(summary['train'].values()):7d} "
          f"{sum(summary['val'].values()):7d} {sum(summary['test'].values()):7d}  "
          f"{grand_total:7d}")


def main():
    parser = argparse.ArgumentParser(
        description="Chuẩn bị dataset từ E:\\bb60c\\captures\\",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--src",   default=r"E:\bb60c\captures",
                        help="Thư mục gốc chứa no_uav/ và uav/")
    parser.add_argument("--dst",   default=r"E:/bb60c/dataset",
                        help="Thư mục đích")
    parser.add_argument("--split", nargs=3, type=float,
                        default=[0.70, 0.15, 0.15],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--mode",  choices=["both", "binary", "multiclass"],
                        default="both",
                        help="Tạo dataset nào:\n"
                             "  binary     → 2 lớp: uav / no_uav\n"
                             "  multiclass → 8 lớp: mỗi loại riêng\n"
                             "  both       → tạo cả hai (mặc định)")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    print(f"\n{'='*62}")
    print(f"  CHUẨN BỊ DATASET UAV WATERFALL")
    print(f"{'='*62}")
    print(f"  Nguồn : {src}")
    print(f"  Đích  : {dst}")
    print(f"  Split : train={args.split[0]:.0%}  val={args.split[1]:.0%}  test={args.split[2]:.0%}")
    print(f"  Mode  : {args.mode}")

    if not src.exists():
        print(f"\n  [!] Thư mục nguồn không tồn tại: {src}")
        return

    assert abs(sum(args.split) - 1.0) < 1e-6, "Tổng split phải = 1.0"

    # Quét tất cả ảnh
    print(f"\n  Đang quét ảnh trong {src}...")
    records = scan_source(src)
    print(f"  Tìm thấy {len(records)} ảnh PNG có nhãn")

    if not records:
        print(f"\n  [!] Không tìm thấy ảnh. Kiểm tra lại đường dẫn và tên thư mục.")
        print(f"  Các thư mục nhận diện được: {list(FOLDER_TO_CLASS.keys())}")
        return

    # Thêm nhãn binary
    for r in records:
        r["binary_label"] = "uav" if r["label"] in UAV_CLASSES else "no_uav"

    # ── Dataset BINARY ────────────────────────────────────────────────────────
    if args.mode in ("binary", "both"):
        dst_bin = dst / "binary"
        print(f"\n{'─'*62}")
        print(f"  [A] BINARY dataset → {dst_bin}")
        print(f"      Lớp: uav / no_uav")
        split_and_copy(records, dst_bin, tuple(args.split), "binary_label", args.seed)

    # ── Dataset MULTICLASS ────────────────────────────────────────────────────
    if args.mode in ("multiclass", "both"):
        dst_mc = dst / "multiclass"
        print(f"\n{'─'*62}")
        print(f"  [B] MULTICLASS dataset → {dst_mc}")
        print(f"      Lớp: mỗi loại tín hiệu riêng")
        split_and_copy(records, dst_mc, tuple(args.split), "label", args.seed)

    # ── Kiểm tra số lượng tối thiểu ───────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  Kiểm tra số lượng mẫu:")
    by_lbl = defaultdict(int)
    for r in records:
        by_lbl[r["label"]] += 1

    ok = True
    for lbl, cnt in sorted(by_lbl.items()):
        tr_cnt = int(cnt * args.split[0])
        status = "✓" if tr_cnt >= 100 else ("⚠ ÍT" if tr_cnt >= 30 else "✗ QUÁ ÍT")
        if tr_cnt < 100: ok = False
        print(f"    {status}  {lbl:30s}: {tr_cnt:4d} ảnh train "
              f"({'đủ' if tr_cnt >= 100 else 'cần thêm'})")

    if not ok:
        print(f"""
  [!] Một số lớp có ít ảnh train (<100). Gợi ý:
      - Tiếp tục thu thêm ảnh cho các lớp thiếu
      - Hoặc dùng --mode binary để ghép tất cả UAV thành 1 lớp
      - Tối thiểu cần 100 ảnh/lớp để model học được
      - Tốt nhất 500+ ảnh/lớp
""")

    print(f"\n  ✓ Dataset sẵn sàng!")
    print(f"  → Tiếp theo:")
    if args.mode in ("binary", "both"):
        print(f"    python step2_train.py --mode classify "
              f"--data {dst / 'binary'} --epochs 50 --batch 32")
    if args.mode in ("multiclass", "both"):
        print(f"    python step2_train.py --mode classify "
              f"--data {dst / 'multiclass'} --epochs 100 --batch 16")
    print()


if __name__ == "__main__":
    main()
